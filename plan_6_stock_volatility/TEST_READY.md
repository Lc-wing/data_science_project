# TEST_READY — Plan 6 Phase A E2E & Adversarial Hardening (M6)

Status: **Phase A Complete & Ready for Final CS482 Delivery / Phase B Handoff**.
This document records the Milestone M6 5-tier opaque-box E2E and white-box single-fault adversarial verification against `origin/main` baseline `63dc8ec46dddd910241de6c5ba968f9de308bcc3` (M5), together with the surgical white-box hardening improvements landed in M6.

No scientific threshold, statistical test, or anti-leakage boundary was relaxed to obtain a pass. All M1–M2 frozen artifacts (`data/frozen/`) remain bit-for-bit unchanged.

## Verification Commands

Run from `plan_6_stock_volatility/` with `uv` on `PATH`:

```bash
cd plan_6_stock_volatility
uv sync --all-groups

# 1. Lint check (zero warnings/errors)
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync ruff check .

# 2. Full 5-tier E2E & adversarial suite (Tiers 1-5)
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync pytest tests/e2e --project-root . -v

# 3. Complete repository test suite (M0-M5 unit/contract tests + M6 E2E Tiers 1-5)
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync pytest -v

# 4. Full deterministic single-command offline Phase A replay (Steps 01 -> 06)
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync python scripts/run_phase_a.py --skip-download
```

Do not set `STOCK_VOL_E2E_IN_SELF_TEST=1` manually; that environment variable is the internal recursion guard for Tier 5 child fault-injection subprocesses.

## Verified Toolchain

| Tool | Version / Role |
| --- | --- |
| `uv` | `0.13.9` (`x86_64-unknown-linux-gnu`) |
| Python | `3.11.18` (locked `.venv` via `uv.lock`) |
| Poppler (`pdfinfo`, `pdftotext`) | `24.02.0` (A4 page geometry & text extraction audit) |
| TeX Live (`pdflatex`, `bibtex`) | `pdfTeX 3.141592653-2.6-1.40.25 (TeX Live 2023/Debian)` (`SOURCE_DATE_EPOCH` deterministic compilation) |

## E2E Suite Breakdown (Tiers 1–5)

| Suite | File | Tests | Scope |
| --- | --- | ---: | --- |
| Tier 1 | `tests/e2e/test_tier1_feature_coverage.py` | 6 | End-to-end feature coverage across all 20 Plan 6 features (M0–M5 artifacts, schemas, provenance, freeze manifest) |
| Tier 2 | `tests/e2e/test_tier2_boundary_corner.py` | 7 | DST/early-close cutoffs, `<=5`/day cap, purged splits, independent Parkinson RV / `y_tail` / Train-only `rv90` / M4 QLIKE-PR-AUC-ECE / Holm recomputation, TreeSHAP additivity `< 1e-6`, 300-DPI PNGs, every-page A4 PDF & zero causal wording |
| Tier 3 | `tests/e2e/test_tier3_cross_feature.py` | 4 | Cross-milestone SHA-256 parity, M1–M4 row-count & `(ticker, date)` 1-to-1 key alignment, Table 1 CSV/MD/LaTeX consistency, all 51 numeric claims verified in PDF & `phase_a_audit.md` |
| Tier 4 | `tests/e2e/test_tier4_real_world_scenarios.py` | 5 | Subprocess `--check`, isolated clone `run_phase_a.py --report-only --project-root` bit-identical SHA-256 parity across all 11 M5 files (including 3 PDFs), frozen tamper abort, raw provenance tamper abort, and M6 doc / gitignore checks |
| Tier 5 | `tests/e2e/test_tier5_adversarial.py` | 9 | 8 named single-fault adversarial mutations injected into isolated clones (each required to fail cleanly with exit `1`, `0` errors, and exact diagnostic clause) + white-box PDF auditor hardening unit test |
| **E2E Total** | `tests/e2e/` | **31** | **0 failed, 0 errors, 0 skipped** |
| **Phase A Total** | `tests/` (M0–M6) | **97** | **66 M0–M5 tests + 31 M6 E2E/adversarial tests (0 failed, 0 skipped)** |
| **Full Repo Total** | `tests/` (M0–M8) | **104** | **97 Phase A tests + 7 Phase B (`tests/test_phase_b.py`) tests (0 failed, 0 skipped)** |

## Tier 5 Named Single-Fault Adversarial Matrix

| Fault ID | Injected Fault (Isolated Clone) | Expected Diagnostic Clause |
| --- | --- | --- |
| `freeze_manifest_sha_tamper` | Append 1 byte to `data/frozen/analysis_spec.json` | `SHA-256 mismatch` (2 failed in Tier 1 & Tier 3) |
| `event_cap_violation` | Mutate 1 row of `events_text_phase_b.parquet` to `news_count=6` | `Event cap violated` (Tier 2) |
| `split_embargo_leak` | Relabel an `embargo_1` date as `val` in `panel_daily.parquet` | `Val boundary breach` (Tier 2) |
| `test_tail_target_leak` | Unmask `has_full_target=True` on `2025-12-31` in `panel_daily.parquet` | `Incomplete forward target leakage` (Tier 2) |
| `table1_numeric_drift` | Replace `0.2306` with `0.1999` in `outputs/tables/table1_summary.tex` | `missing from Table 1 LaTeX` (Tier 3) |
| `pdf_page2_us_letter` | Replace PDF Page 2 mediabox with US Letter (`612x792 pts`) while Page 1 stays A4 | `page 2 is not A4` (Tier 2 & `audit_compiled_pdf`) |
| `figure_low_dpi_150` | Re-save `fig1_volatility_regimes_and_forecasts.png` at `150 DPI` | `DPI below 300` (Tier 2) |
| `claim_dropped_from_audit` | Alter `LGBM_QLIKE` in `report/phase_a_audit.md` | `Claims missing from phase_a_audit.md` & `LGBM_QLIKE` (Tier 3) |

## Forensic Spot Check (Committed Tree)

- `data/frozen/FREEZE_MANIFEST.json` SHA-256: `8aa3b379b8beef2b077d5c3e6651733be089eea45e0ac56b738de07a30aa3b1b` (all 6 frozen files match on-disk SHA-256).
- Full offline replay (`scripts/run_phase_a.py --skip-download`) completed Steps `01 -> 06` with exit `0` and `0` modified tracked files (`git status --short` completely clean — 100% bit-identical across all Parquet, CSV, JSON, PNG, and PDF artifacts).
- `report/CS482_GroupX_StockVolatilityLLM_PhaseA_Draft.pdf` SHA-256: `879c5e6a7b8ce2a429ded418bc52a60c55346f88b378a20e2d8510eec3eb6e4b` (7 pages, every page exact A4 `595.28 x 841.89 pts`, `4.00` A4 main-text pages, `51/51` claims verified, `0` forbidden causal words or unresolved LaTeX artifacts).
- Sibling workspaces (`plan_2_tourism_hotel/`, `plan_5_spatial_equity/`, `plans/`) remain 100% untouched (`0` diffs).
