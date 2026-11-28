# Milestone M3: Cross-Cycle EDA & H1 Stock-FE Extended Panel HAR-RV (里程碑 M3：跨周期 EDA 与 H1 面板扩展 HAR-RV 回归)

- **Milestone**: `M3 — Cross-Cycle EDA & H1 Stock-FE Extended Panel HAR-RV (跨周期探索性数据分析与 H1 股票固定效应面板扩展 HAR-RV 回归)`
- **Status**: `COMPLETED & PUSHED`
- **Parent / Baseline Commit SHA**: `3c9bba99b09c53a838d787a60b0b7ebb9c4fb025` (`M2`)
- **Date (UTC)**: `2026-11-23`
- **Frozen Source Provenance Timestamp (UTC)**: `2026-11-22T14:04:28+00:00`
- **Verification Verdict**: `INDEPENDENT VERIFICATION PASS` (`ruff check .`: 0 errors; `pytest -v`: `51 passed` [26 M0 + 9 M1 + 8 M2 + 8 M3]; `03_eda.py` & `04_panel_har_rv.py` verified deterministic offline from `FREEZE_MANIFEST.json`; Driscoll–Kraay SE verified numerically against `statsmodels` `hac-groupsum`; `git check-ignore` exit `1`; `0` sibling diffs; `0` secrets)

---

## 1. Milestone Objectives & Scope (里程碑目标与范围)

Milestone M3 executes strictly offline on M2's cryptographically verified `data/frozen/` bundle (`verify_freeze_manifest()` called at startup of every script):

1. **Cross-Cycle Exploratory Data Analysis (`src/stock_volatility/eda.py`, `scripts/03_eda.py`)**:
   - Quantify 2019–2025 volatility regimes (2019 pre-COVID, 2020 COVID shock, 2021 rebound, 2022 rate-hike bear market, 2023 disinflation, 2024 AI validation cycle, 2025 AI/policy test cycle, and full sample) across `ALL` and all 5 GICS sectors.
   - Measure leverage asymmetry across negative vs non-negative and 6-bin lagged daily return buckets (`ret_1d <= -3%` through `>= +3%`).
   - Audit 2019–2023 vs 2024–2025 aligned corporate-event coverage consistency after M2's 16:00/13:00 ET cutoff, dedup, and `<= 5`/day cap.
   - Contrast downside tail-risk incidence (`y_tail` and `y_tail_rv90`) across Train-median High/Low VIX (`VIX >= 19.6100`, fitted strictly on 2019–2023 Train daily median) × Event status (`No_Event`, `Has_Event`, `Has_8K_Earnings`) with explicit numerators, denominators, and Wilson 95% CIs.
2. **H1 Stock-FE Extended Panel HAR-RV (`src/stock_volatility/regression.py`, `scripts/04_panel_har_rv.py`)**:
   - Estimate the within-entity Stock-FE extended Panel HAR-RV model (`log_rv_d`, `log_rv_w`, `log_rv_m`, `neg_ret_1d`, `vix_close`, plus `Healthcare`-reference sector interactions for `Technology`, `Financials`, `Consumer`, `Energy`) with Bartlett-kernel **Driscoll–Kraay SE** (`L = max(5, floor(4 * (T/100)^(2/9))) = 7`).
   - Test the 5 pre-registered confirmatory H1 hypotheses with step-down **Holm–Bonferroni** adjustment and report all 3 pre-registered sensitivities (`train_only_2019_2023`, `close_to_close_rv_5d_cc_fwd`, `newey_west_entity_clustered_se`) plus 2 supplementary checks side-by-side (zero pretest switching, C15 compliant).

---

## 2. Milestone Deliverables & File Inventory (交付物清单与文件契约)

### 2.1 Source Code, CLIs, Tests & Documentation

| File Path (Relative to `plan_6_stock_volatility/`) | Category | Role & Interface Contract |
| --- | --- | --- |
| `src/stock_volatility/eda.py` | Source | `compute_regimes_by_sector()`, `compute_leverage_asymmetry()`, `compute_coverage_consistency()`, `compute_vix_news_tail_incidence()`, `render_eda_summary_markdown()`, `run_eda_pipeline()` |
| `src/stock_volatility/regression.py` | Source | `driscoll_kraay_bandwidth()`, `holm_bonferroni_adjust()`, `fit_stock_fe_driscoll_kraay()` (`driscoll_kraay` & `newey_west_entity`), `render_h1_summary_markdown()`, `run_h1_regression_pipeline()` |
| `scripts/03_eda.py` | CLI | Verifies `FREEZE_MANIFEST.json` and deterministically emits all 6 EDA artifacts |
| `scripts/04_panel_har_rv.py` | CLI | Verifies `FREEZE_MANIFEST.json` and deterministically emits all 4 H1 regression artifacts |
| `tests/test_eda.py` | Test | 3 tests covering pure EDA math & Train-only VIX median, offline determinism, committed EDA contracts, and zero forbidden causal wording |
| `tests/test_regression.py` | Test | 5 tests covering exact Holm–Bonferroni math, numerical equivalence of `fit_stock_fe_driscoll_kraay` vs `statsmodels` `hac-groupsum`, offline determinism, committed H1 contracts, and M3 milestone doc verification |
| `docs/milestones/M3_eda_and_h1_panel_har_rv.md` | Doc | This milestone delivery record & AI execution log (plus ledger updates in `docs/milestones/README.md`, `plan_6_stock_volatility/README.md`, and root `README.md`) |

### 2.2 Committed M3 Outputs & Cryptographic SHA-256 Digests

| File Path (Relative to `plan_6_stock_volatility/`) | Byte Size | SHA-256 Digest | Content Summary |
| --- | ---: | --- | --- |
| `outputs/tables/eda_regimes_by_sector.csv` | `11,296 B` | `9c2c6aae5cd1c6e5fdc45d1ad2ca9763146d2861650123997d1ca8cb5556ca24` | 8 regimes × 6 sector slices (`ALL` + 5 sectors) |
| `outputs/tables/eda_leverage_asymmetry.csv` | `30,218 B` | `c4e9b574781f5aac5631caef381c1649fda1dfdf9442fadbddc18edcc00d6503` | 4 scopes × 6 sectors × 8 lagged return buckets |
| `outputs/tables/eda_coverage_consistency.csv` | `1,302 B` | `64bf6c7783201a58c433bff418679e3c0f5052a9c0a942ed68f2b6c31114826a` | 2019–2023 vs 2024–2025 aligned event-day & per-year rates (`60/60` active) |
| `outputs/tables/eda_vix_news_tail_incidence.csv` | `4,339 B` | `8535a56efecbddcbf4d3cef1b8fa57f012dd17b38a714071ec91dbd65a89be2e` | High/Low Train-median VIX × event status with `k/N` and Wilson 95% CIs |
| `outputs/tables/eda_summary.md` | `3,832 B` | `09ca81346a36888c72c0cc98e46dbb2aec3575498e949266691733429340c827` | Non-causal descriptive EDA summary |
| `outputs/metrics/eda_summary.json` | `5,730 B` | `e550d8c67852bba034d9e03e4740c609b6ae69bc470930d6a139913255944d5d` | Machine-readable EDA summary metrics for M5 report injection |
| `outputs/tables/h1_panel_har_rv_main.csv` | `4,178 B` | `f9d83f93cc2b5aa38e383836f3eedb174e56062629f37856ad76b0421fa85ce8` | 13 main regressors, 2 confirmatory 5-sector AMEs, 10 sector total slopes |
| `outputs/tables/h1_panel_har_rv_sensitivities.csv` | `6,729 B` | `e533b205f9a3f39456210ea526a324f29bb5f4225d9d87e8644de3636c23cf55` | 3 pre-registered + 2 supplementary robustness specifications |
| `outputs/tables/h1_summary.md` | `6,229 B` | `6185a5f86988d483797dfad00585caa5eb977b942f1a9efe6627b9ac3d1bae4e` | Associational H1 summary with Holm-adjusted family and sensitivities |
| `outputs/metrics/h1_regression.json` | `12,512 B` | `d1faf618d5c1a312f89827f49bfaa4077855146454e6d02717d4deefeca7e5cf` | Machine-readable H1 coefficients, Holm CIs, Wald stats & sensitivities |

---

## 3. Technical Implementation & Key Empirical Findings (核心技术实现与关键实证结果)

### 3.1 EDA Highlights (`104,040` Modeling-Complete Stock-Days, `1,734` Days)

- **Cross-Cycle Volatility Regimes (`ALL`)**: Mean 5-day annualized Parkinson RV spiked during `covid_shock_2020` (`0.3241`, `VIX = 29.25`, `y_tail = 13.55%` [`2,057/15,180`]) and `rate_hike_bear_2022` (`0.2687`, `VIX = 25.62`, `y_tail = 12.68%` [`1,910/15,060`]), compressed in `ai_cycle_val_2024` (`0.1998`, `VIX = 15.61`, `y_tail = 6.08%`), and re-expanded in `ai_cycle_test_2025` (`0.2411`, `VIX = 19.06`, `y_tail = 8.62%` [`1,267/14,700`]).
- **Leverage Asymmetry**: Mean forward RV after negative daily returns (`ret_1d < 0`) is `0.2406` vs `0.2268` after `ret_1d >= 0` (`1.061×`); for extreme moves (`ret_1d <= -3%` vs `ret_1d >= +3%`), `0.3996` vs `0.3518` (`1.136×`).
- **Coverage Consistency (Post-Cutoff & Dedup)**: `60/60` active tickers in both `2019–2023` (`5,040/75,480` = `6.68%` event days) and `2024–2025` (`1,782/30,120` = `5.92%` event days, rate ratio `0.8860`, no structural jump).
- **VIX Regime × Event-Day Tail Incidence (Train Median `VIX >= 19.6100`)**:
  - `Low_VIX`: `No_Event = 6.64%` (`3,810/57,411`), `Has_Event = 7.51%` (`289/3,849`), `Has_8K_Earnings = 8.17%` (`83/1,016`).
  - `High_VIX`: `No_Event = 10.61%` (`4,235/39,914`), `Has_Event = 10.05%` (`288/2,866`), with extreme volatility `y_tail_rv90` higher on `Has_Event` (`18.91%` vs `16.62%`).

### 3.2 H1 Stock-FE Extended Panel HAR-RV Results (`Driscoll–Kraay SE, L = 7`, Within $R^2 = 0.4736$)

| Confirmatory Test ID | Estimate / Stat | DK SE | `t` / `F` | Raw `p` | Holm `p` | 95% CI | Holm Sig (`α=0.05`) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `neg_ret_1d` *(5-sector AME)* | `-1.44577` | `0.30956` | `-4.67` | `3.237e-06` | `9.711e-06` | [`-2.0529`, `-0.8386`] | **YES** |
| `vix_close` *(5-sector AME)* | `+0.01411` | `0.00165` | `+8.55` | `2.595e-17` | `1.298e-16` | [`+0.0109`, `+0.0173`] | **YES** |
| `tech_x_neg_ret_1d_vs_healthcare` | `-0.64789` | `0.37673` | `-1.72` | `8.566e-02` | `1.713e-01` | [`-1.3868`, `+0.0910`] | **NO** |
| `tech_x_vix_close_vs_healthcare` | `+0.00090` | `0.00103` | `+0.88` | `3.797e-01` | `3.797e-01` | [`-0.0011`, `+0.0029`] | **NO** |
| `joint_5sector_interaction_wald` | `χ² = 84.835` | — | `F(8,1733)=10.60` | `1.234e-14` | `4.935e-14` | — | **YES** |

- **Honest H1 Verdict (Partial Support)**: Overall leverage (`-1.4458`, meaning negative returns strongly predict higher forward log-RV), overall VIX (`+0.01411`), and joint 5-sector heterogeneity (`Wald χ² = 84.84`, `p_holm = 4.94e-14`) are overwhelmingly significant across every sensitivity (`train_only_2019_2023`, `close_to_close_rv_5d_cc_fwd`, `extended_with_macro_and_events`). Cross-sector heterogeneity is led by Financials (leverage `-1.273`, VIX `+0.00504`), Energy (VIX `+0.00874`), and Consumer (VIX `+0.00367`) relative to defensive Healthcare, while the specific pairwise `Technology vs Healthcare` contrast is directionally negative on leverage (`-0.648`) but not individually significant under conservative Driscoll–Kraay SEs and Holm correction.

---

## 4. What the AI Agents Did in Milestone M3 (本里程碑内 AI 完成的工作全记录)

### 4.1 Pre-Implementation Econometric Verification & Design

- Verified exact equivalence of our Stock-FE Bartlett Driscoll–Kraay covariance estimator against `statsmodels.OLS(..., cov_type="hac-groupsum")` on demeaned data (`ratio = 1.00064`, matching the exact `sqrt((N - K)/(N - K - 60))` absorbed fixed-effects degree-of-freedom factor) and locked this cross-check permanently inside `tests/test_regression.py`.
- Designed the H1 confirmatory family around exact linear contrasts (`ame_5sector_neg_ret_1d`, `ame_5sector_vix_close`, `Technology vs Healthcare` differentials, and the `df = 8` joint sector-interaction Wald test) so all 5 Holm-adjusted tests derive from a single unified Driscoll–Kraay covariance matrix.

### 4.2 Traps Caught & Resolved

1. **Cross-Sectional Correlation Understatement in Entity-Clustered SEs**: Comparing `driscoll_kraay` vs `newey_west_entity_clustered_se` empirically demonstrated that ignoring contemporaneous market shocks shrinks SEs by `3.6×–4.7×` and creates a false nominal `p = 0.028` on `tech_x_neg_ret_1d_vs_healthcare` that disappears (`p = 0.086`, `p_holm = 0.171`) under Driscoll–Kraay SEs.
2. **JSON `None` vs `NaN` Formatting Bug**: During initial execution of `scripts/04_panel_har_rv.py`, `_clean_json()` normalized Wald-test `NaN` SE/CI fields to `None` before rendering `h1_summary.md`, triggering a `TypeError` in `math.isnan(None)`. Fixed immediately in `render_h1_summary_markdown()` by explicitly guarding `None` alongside `math.isnan()`.

### 4.3 TDD & Independent Verification (`test-driven-development`, `verification-before-completion`)

1. **RED Phase**: Authored `tests/test_eda.py` (`3` tests) and `tests/test_regression.py` (`5` tests) first; confirmed `8 failed in 1.94s`.
2. **GREEN Phase**: Implemented `eda.py`, `regression.py`, `03_eda.py`, and `04_panel_har_rv.py`; resolved the `None` formatting bug and 3 `ruff` E501 line-length items; generated all 10 M3 outputs.
3. **Final Verification**: Verified `ruff check .` (`0` errors), full `pytest -v` (`51 passed`), fresh CLI replays of `03_eda.py` and `04_panel_har_rv.py`, `git check-ignore` exit `1`, zero sibling diffs, and zero secrets prior to GitHub push.

---

## 5. Verification Commands & Observed Results (独立验证命令与实测输出)

```bash
cd /home/antigravity/Lc-wing/data_science_project/plan_6_stock_volatility
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync python scripts/03_eda.py
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync python scripts/04_panel_har_rv.py
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync ruff check .
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync pytest -v
```

---

## 6. Git Hygiene & Delivery Checklist (Git 交付检查清单)

- [x] Dual-content record complete (M3 deliverables, SHA-256 digests & empirical findings + AI execution log & traps resolved)
- [x] `docs/milestones/README.md`, `plan_6_stock_volatility/README.md`, and root `README.md` updated and linked
- [x] `git check-ignore -v --no-index plan_6_stock_volatility/docs/milestones/*.md` exited `1`
- [x] Zero sibling diffs in `plan_2_tourism_hotel/`, `plan_5_spatial_equity/`, and `plans/`
- [x] All staged files `< 25 MB`, zero `.agents/` or secret leakage
