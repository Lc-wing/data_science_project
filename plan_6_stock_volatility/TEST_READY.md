# TEST_READY — Unified Verification, 5-Tier E2E & Adversarial Hardening Guide

Status: **Unified Repository & Final CS482 Submission Verified (`104/104` Pytest Passed, `61/61` PDF Claims Audited)**.

This document specifies the automated verification, deterministic offline replay, and single-fault white-box adversarial test suite covering the unified `plan_6_stock_volatility/` codebase, frozen dataset (`data/frozen/`), unified outputs (`outputs/{figures,metrics,predictions,tables}/`), and single authoritative CS482 report (`report/CS482_GroupX_StockVolatilityLLM.pdf`).

No scientific threshold, statistical test, or anti-leakage boundary is relaxed in any test. All frozen inputs under `data/frozen/` remain 100% bit-for-bit invariant (`FREEZE_MANIFEST.json` SHA-256 `8aa3b379b8beef2b077d5c3e6651733be089eea45e0ac56b738de07a30aa3b1b`).

---

## 1. Verification Commands

Run from `plan_6_stock_volatility/` with `uv` on `PATH`:

```bash
cd plan_6_stock_volatility
uv sync --all-groups

# 1. Static lint check (zero warnings/errors across src/, scripts/, tests/)
uv run --no-sync ruff check .

# 2. Standalone 5-tier opaque-box E2E & white-box adversarial suite (31 tests)
uv run --no-sync pytest tests/e2e --project-root . -v

# 3. Complete 104-test repository suite (73 unit/contract/DL tests + 31 E2E/adversarial tests)
uv run --no-sync pytest -v

# 4. Fast report-only deterministic rebuild & 61-claim PDF audit
uv run --no-sync python scripts/run_pipeline.py --report-only

# 5. Complete offline CPU pipeline replay (Steps 01 -> 02 -> 03 -> 04 -> 05 -> 07; zero network calls)
uv run --no-sync python scripts/run_pipeline.py --skip-download
```

> **Note**: Do not set `STOCK_VOL_E2E_IN_SELF_TEST=1` manually; that environment variable is the internal recursion guard used by Tier 5 when spawning isolated child fault-injection subprocesses. `stock_volatility/__init__.py` automatically sets single-thread BLAS/OpenMP defaults (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) and `MPLBACKEND=Agg` at import time.

---

## 2. Verified Toolchain

| Tool | Version / Role |
| --- | --- |
| `uv` | Locked Python dependency & virtual environment manager (`uv.lock`) |
| Python | `3.11+` (`.venv` with `numpy`, `pandas`, `statsmodels`, `lightgbm`, `arch`, `shap`, `torch`) |
| Poppler (`pdfinfo`, `pdftotext`) | Every-page A4 geometry (`595.28 x 841.89 pts`) & full-text claim extraction audit |
| TeX Live (`pdflatex`, `bibtex`) | Deterministic `SOURCE_DATE_EPOCH` PDF compilation (`report/CS482_GroupX_StockVolatilityLLM.pdf`) |

---

## 3. Repository Test Architecture (`104` Total Tests)

### 3.1 Module & Contract Test Suites (`73` Tests)

| File | Tests | Scope Aligned with CS482 Report |
| --- | ---: | --- |
| `tests/test_smoke.py` | 26 | Package metadata, canonical `outputs/{tables,metrics,figures,predictions}` directories, `UNIVERSE_60` invariants, split chronology, library imports & LaTeX/Poppler smoke compile |
| `tests/test_ingest.py` | 9 | Provenance SHA-256 verification, Yahoo OHLCV, backward-only FRED macro alignment, paginated SEC EDGAR (`BLK`/`XOM` predecessor CIKs) & offline `--skip-download` determinism |
| `tests/test_panel_and_freeze.py` | 8 | DST-aware `16:00 ET` & `13:00 ET` early-close cutoffs, intra-day deduplication & `<= 5`/day cap, strictly `<= t` features, `6`-day purged embargoes & `FREEZE_MANIFEST.json` |
| `tests/test_eda.py` | 3 | Cross-cycle volatility regimes, Train-only VIX median threshold, leverage asymmetry & non-causal wording checks |
| `tests/test_regression.py` | 4 | Exact equivalence of Stock-FE Bartlett Driscoll–Kraay SE against `statsmodels` (`hac-groupsum` up to finite-sample FE dof), Holm step-down & H1 sensitivity contracts |
| `tests/test_forecast.py` | 4 | Quarterly expanding `>= 5`-day purged folds, Duan smearing, closed-form GJR-GARCH recursion, OOF probability calibrators & `outputs/predictions/h2_*_predictions.parquet` |
| `tests/test_evaluation_and_shap.py` | 5 | QLIKE/ECE mathematical properties, `h=5` Bartlett-PSD HLN-DM test, date-block bootstrap determinism & Train-background exact TreeSHAP additivity (`< 1e-7`) |
| `tests/test_figures_and_report.py` | 7 | 300-DPI PNG `pHYs` & vector PDF checks, Table 1 builder, forbidden causal word scanner, A4 `[3.0, 5.0]` page-budget check & `scripts/run_pipeline.py --report-only` |
| `tests/test_dl_and_llm.py` | 7 | `analysis/dl/` 60-day sequence windows, zero-leakage prompts, PyTorch `PatchTST` (`1M`/`15M` params, non-crossing Pinball quantiles), `Gemma-4`/`Qwen3.5` text-only LoRA filter, 2024-Val OOF meta-fusion, `06_dl_and_llm_gpu.py` GPU guards, H3 Table 2 & 61-claim final PDF |

### 3.2 5-Tier Opaque-Box E2E & White-Box Adversarial Suite (`31` Tests)

| Suite | File | Tests | Scope |
| --- | --- | ---: | --- |
| Tier 1 | `tests/e2e/test_tier1_feature_coverage.py` | 6 | Complete artifact & schema coverage across raw provenance, frozen bundle, EDA/H1, H2 `outputs/predictions/`, H3 Table 2 (`table2_h3_ablation.*`, `h3_dl_summary.json`) & unified `61/61` PDF audit |
| Tier 2 | `tests/e2e/test_tier2_boundary_corner.py` | 7 | DST/early-close cutoffs, `<= 5`/day cap, purged splits, independent Parkinson RV / `y_tail` / Train-only `rv90` / QLIKE-PR-AUC-ECE / Holm recomputation, TreeSHAP `< 1e-6`, 300-DPI PNGs & every-page A4 PDF |
| Tier 3 | `tests/e2e/test_tier3_cross_feature.py` | 4 | Cross-stage SHA-256 parity (`report/report.tex` & PDF cover), `105,600` / `104,040` / `14,700` `(ticker, date)` 1-to-1 alignment, Table 1 CSV/MD/LaTeX parity & all `61` claims traced in PDF + `report/report_audit.md` |
| Tier 4 | `tests/e2e/test_tier4_real_world_scenarios.py` | 5 | Subprocess `--check`, isolated clone `scripts/run_pipeline.py --report-only --project-root` bit-for-bit SHA-256 identity across all 11 report/figure files, frozen tamper abort & raw provenance tamper abort |
| Tier 5 | `tests/e2e/test_tier5_adversarial.py` | 9 | 8 named single-fault mutations injected into isolated clones (each required to fail with exit `1`, `0` errors & exact diagnostic clause) + white-box PDF auditor US-Letter / `TBD_NUM` unit check |
| **Total** | **`tests/`** | **104** | **`104 passed`, `0 failed`, `0 errors`, `0 skipped`** |

---

## 4. Tier 5 Named Single-Fault Adversarial Matrix

| Fault ID | Injected Fault (Isolated Clone) | Expected Diagnostic Clause |
| --- | --- | --- |
| `freeze_manifest_sha_tamper` | Append 1 byte to `data/frozen/analysis_spec.json` | `SHA-256 mismatch` (2 failed in Tier 1 & Tier 3) |
| `event_cap_violation` | Mutate 1 row of `events_text_phase_b.parquet` to `news_count=6` | `Event cap violated` (Tier 2) |
| `split_embargo_leak` | Relabel an `embargo_1` date as `val` in `panel_daily.parquet` | `Val boundary breach` (Tier 2) |
| `test_tail_target_leak` | Unmask `has_full_target=True` on `2025-12-31` in `panel_daily.parquet` | `Incomplete forward target leakage` (Tier 2) |
| `table1_numeric_drift` | Replace `0.2306` with `0.1999` in `outputs/tables/table1_summary.tex` | `missing from Table 1 LaTeX` (Tier 3) |
| `pdf_page2_us_letter` | Replace PDF Page 2 mediabox with US Letter (`612x792 pts`) while Page 1 stays A4 | `page 2 is not A4` (Tier 2 & `audit_compiled_pdf`) |
| `figure_low_dpi_150` | Re-save `fig1_volatility_regimes_and_forecasts.png` at `150 DPI` | `DPI below 300` (Tier 2) |
| `claim_dropped_from_audit` | Alter `LGBM_QLIKE` in `report/report_audit.md` | `Claims missing from report_audit.md` & `LGBM_QLIKE` (Tier 3) |
