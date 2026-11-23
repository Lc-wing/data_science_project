# Milestone M6: Opaque-Box E2E Verification (Tiers 1–4), White-Box Adversarial Hardening (Tier 5) & Final Phase A Delivery Sign-Off (里程碑 M6：端到端黑盒测试、白盒对抗性加固与阶段 A 最终交付验收)

- **Milestone**: `M6 — Opaque-Box E2E Tiers 1–4, Tier 5 Adversarial Hardening & Final Phase A Sign-Off (端到端四层黑盒验证、第五层白盒单点故障对抗加固与阶段 A 最终交付验收)`
- **Status**: `COMPLETED & VERIFIED (FINAL PHASE A MILESTONE)`
- **Parent / Baseline Commit SHA**: `63dc8ec46dddd910241de6c5ba968f9de308bcc3` (`M5` on `origin/main`)
- **Date (UTC)**: `2026-11-23`
- **Verification Verdict**: `VERIFIED PASS` (`ruff check .`: 0 errors; `pytest tests/e2e --project-root . -v`: `31/31` passed; full `pytest -v`: `97/97` passed [66 M0–M5 + 31 M6 E2E/adversarial]; full offline `scripts/run_phase_a.py --skip-download`: exit `0` with `0` byte diffs across all M1–M5 files; `git check-ignore`: exit `1`; `0` sibling diffs; `0` secrets)

---

## 1. Milestone Objectives & Scope (里程碑目标与范围)

Milestone M6 is the final verification and hardening gate for Plan 6 Phase A (`plan_6_stock_volatility/`):

1. **Full Offline Single-Command Replay Proof (`scripts/run_phase_a.py --skip-download`)**:
   - Execute the complete 6-step Phase A pipeline (`01_download_and_audit --skip-download` → `02` SHA-256 freeze check → `03` EDA → `04` H1 Panel HAR-RV → `05` 300 GJR-GARCH fits + Global LightGBM + 2024 Val OOF Calibration + 2025 Holdout + 1,000-rep date-block bootstrap + `h=5` HLN-DM/Holm + Train-background TreeSHAP → `06` 300-DPI Figure 1/2 + Table 1 + deterministic `SOURCE_DATE_EPOCH` LaTeX/PDF + 51-claim audit) offline and prove zero drift against committed artifacts (`git status --short` completely clean).
2. **Opaque-Box E2E Suite — Tiers 1–4 (`tests/e2e/`)**:
   - **Tier 1 (`test_tier1_feature_coverage.py`)**: End-to-end feature coverage across all 20 Plan 6 features (M0–M5).
   - **Tier 2 (`test_tier2_boundary_corner.py`)**: Independent mathematical recomputation of Parkinson 5-day RV, `y_tail`, Train-only `rv90_threshold`, `neg_ret_1d`, M4 QLIKE/RMSE/MAE/PR-AUC/ROC-AUC/Brier/ECE, Holm monotonicity across H1 (5 tests) and H2 HLN-DM (9 tests), DST/early-close `16:00/13:00 ET` cutoffs, `<= 5`/day cap, purged+embargoed splits, final-5-day target masking, TreeSHAP additivity (`< 1e-6`), 300-DPI PNG metadata, and per-page A4 PDF & causal-wording boundaries.
   - **Tier 3 (`test_tier3_cross_feature.py`)**: Cross-milestone SHA-256 provenance, M1–M4 row count & `(ticker, date)` 1-to-1 key alignment, Table 1 CSV/Markdown/LaTeX consistency, and all 51 numeric claims cross-checked between JSON/CSV sources, `phase_a_audit.md`, and extracted PDF text.
   - **Tier 4 (`test_tier4_real_world_scenarios.py`)**: Subprocess CLI scenarios (`02_build_panel_and_freeze.py --check`, isolated-clone `run_phase_a.py --report-only --project-root` bit-identical SHA-256 parity across all 11 M5 files including all 3 PDFs, tampered frozen manifest abort, tampered raw provenance abort, and M6 delivery doc + `TEST_READY.md` git-tracking checks).
3. **White-Box Adversarial Hardening — Tier 5 (`test_tier5_adversarial.py`) & Targeted Product Fixes**:
   - Hunt and close real white-box edge-case gaps in `report.py`, `ingest/audit.py`, and `scripts/run_phase_a.py`, then verify **8 named single-fault adversarial mutations** injected into isolated temporary project clones (never mutating the committed workspace), requiring each fault to exit cleanly (`returncode == 1`, `errors == 0`) with its exact named diagnostic clause.
4. **Final Delivery Inspection & Sign-Off Artifacts**:
   - Publish `plan_6_stock_volatility/TEST_READY.md` (mirroring `plan_5_spatial_equity/TEST_READY.md`), complete this final M6 milestone record, update all three READMEs/ledgers (`docs/milestones/README.md`, `plan_6_stock_volatility/README.md`, root `README.md`), verify zero sibling or frozen-data changes, and push M6 to GitHub `origin/main`.

---

## 2. Milestone Deliverables & File Inventory (交付物清单与文件契约)

| File Path (Relative to `plan_6_stock_volatility/`) | Category | Size | Contract / Purpose Summary |
| --- | --- | ---: | --- |
| `src/stock_volatility/report.py` | Code (Hardened) | 56,403 B | Hardened `audit_compiled_pdf()` (per-page A4 mediabox check across all pages, `non_a4_pages` return field, multi-page `in_decl` declaration causal-word scanner, `\bTBD_[A-Z0-9_]+\b` placeholder rule) |
| `src/stock_volatility/ingest/audit.py` | Code (Hardened) | 37,254 B | Added backward-compatible `project_root: Path \| None = None` to `run_skip_download_pipeline()` for isolated clone replay & tamper testing |
| `scripts/run_phase_a.py` | CLI (Hardened) | 3,594 B | Added optional `--project-root` CLI flag forwarding `project_root` cleanly to all 6 Phase A steps (`01`–`06`) |
| `tests/conftest.py` | Test | 1,849 B | Added top-level `--project-root` pytest CLI option and `project_root` session fixture (plus `STOCK_VOL_PROJECT_ROOT` env fallback) |
| `tests/e2e/conftest.py` | E2E Test | 3,824 B | Session fixtures for frozen/M4 artifacts and fast isolated `clone_project` fixture (hardlinks read-only raw snapshots, copies mutable trees) |
| `tests/e2e/test_tier1_feature_coverage.py` | E2E Test | 9,224 B | 6 Tier 1 tests covering all 20 Plan 6 Phase A features (M0–M5) |
| `tests/e2e/test_tier2_boundary_corner.py` | E2E Test | 12,338 B | 7 Tier 2 tests for anti-leakage, independent math recomputation, 300-DPI PNGs, every-page A4 PDF & causal wording |
| `tests/e2e/test_tier3_cross_feature.py` | E2E Test | 6,444 B | 4 Tier 3 tests for cross-milestone SHA-256, key alignment, Table 1 CSV/MD/TEX parity & 51-claim PDF audit |
| `tests/e2e/test_tier4_real_world_scenarios.py` | E2E Test | 7,340 B | 5 Tier 4 real-world CLI replay, bit-identical clone parity, tamper defense & M6 doc checks |
| `tests/e2e/test_tier5_adversarial.py` | E2E Test | 9,402 B | 9 Tier 5 tests (8 parameterized single-fault adversarial clone mutations + 1 direct auditor hardening unit test) |
| `TEST_READY.md` | Doc | — | Standalone Phase A E2E & adversarial verification sign-off guide (toolchain, commands, Tier 1–5 counts, Tier 5 fault matrix, forensic spot check) |
| `docs/milestones/M6_e2e_and_adversarial_hardening.md` | Doc | — | This final M6 delivery & AI execution record |
| `docs/milestones/README.md`, `README.md`, `../README.md` | Doc | — | Final ledger marking all milestones `M0`–`M6` complete (`M5` SHA `63dc8ec46dddd910241de6c5ba968f9de308bcc3`) |

---

## 3. Technical Verification & Phase A Final Sign-Off Summary (最终验收指标总览)

### 3.1 Full Phase A Offline Replay Verification (`run_phase_a.py --skip-download`)

Executed full 6-step replay offline from committed M1–M2 snapshots before M6 test authoring:
- **Step 01 (`ingest`)**: `60` stocks, `105,600` OHLCV rows, `1,760` macro days (`13` `T10Y2Y` backward-only ffills), `7,703` paginated SEC EDGAR filings (`60/60` active in both `2019–2023` & `2024–2025`).
- **Step 02 (`freeze`)**: `FREEZE_MANIFEST.json` SHA-256 `8aa3b379b8beef2b077d5c3e6651733be089eea45e0ac56b738de07a30aa3b1b`, `6/6` frozen artifacts verified (`panel_daily.parquet` prefix `3515fcd12cab`).
- **Step 03 (`eda`)**: `104,040` modeling-complete rows verified across all 4 EDA tables.
- **Step 04 (`h1`)**: `N=104,040`, Within `R² = 0.473625`, Bartlett Driscoll–Kraay `L=7`, 5-test Holm family + 5 sensitivities.
- **Step 05 (`h2`)**: `14,760` Val rows, `14,700` Test rows, `300/300` (`100%`) GJR-GARCH(1,1) fits converged, 2024 Val OOF calibrator `isotonic`, Train-background TreeSHAP max additivity errors `4.07e-08` (clf) and `8.35e-08` (reg) `< 1e-6`.
- **Step 06 (`report`)**: `verdict = PASS`, `7` total PDF pages (`1` Cover + `4.00` A4 Main Text + `1` References + `1` Declarations), `51/51` numeric claims verified, `0` forbidden words/artifacts, `SOURCE_DATE_EPOCH` deterministic PDF SHA-256 `879c5e6a7b8ce2a429ded418bc52a60c55346f88b378a20e2d8510eec3eb6e4b`.
- **Post-replay `git status --short`**: **Completely empty (`0` tracked files modified)** — proving 100% end-to-end bit-identical determinism across Parquet, CSV, JSON, PNG, and PDF artifacts.

### 3.2 Complete Plan 6 Phase A Test Coverage (M0–M6)

| Test Module | Milestone | Tests | Result |
| --- | --- | ---: | --- |
| `tests/test_smoke.py` | M0 | 26 | PASS |
| `tests/test_ingest.py` | M1 | 9 | PASS |
| `tests/test_panel_and_freeze.py` | M2 | 8 | PASS |
| `tests/test_eda.py` + `tests/test_regression.py` | M3 | 8 | PASS |
| `tests/test_forecast.py` + `tests/test_evaluation_and_shap.py` | M4 | 9 | PASS |
| `tests/test_figures_and_report.py` | M5 | 6 | PASS |
| `tests/e2e/test_tier1_feature_coverage.py` | M6 (Tier 1) | 6 | PASS |
| `tests/e2e/test_tier2_boundary_corner.py` | M6 (Tier 2) | 7 | PASS |
| `tests/e2e/test_tier3_cross_feature.py` | M6 (Tier 3) | 4 | PASS |
| `tests/e2e/test_tier4_real_world_scenarios.py` | M6 (Tier 4) | 5 | PASS |
| `tests/e2e/test_tier5_adversarial.py` | M6 (Tier 5) | 9 | PASS |
| **Grand Total** | **M0–M6** | **97** | **97 passed, 0 failed, 0 skipped** |

---

## 4. What the AI Agents Did in This Milestone (本里程碑内 AI 完成的工作全记录)

### 4.1 Full-Pipeline Replay & White-Box Adversarial Audit (全流程离线复现与白盒审计)

- Executed the complete 6-step offline replay `scripts/run_phase_a.py --skip-download` end-to-end and verified zero file diffs against the M5 Git commit (`63dc8ec46dddd910241de6c5ba968f9de308bcc3`).
- Audited `src/stock_volatility/report.py`, `src/stock_volatility/panel/freeze.py`, `src/stock_volatility/panel/events.py`, `src/stock_volatility/evaluation.py`, `src/stock_volatility/ingest/audit.py`, and `scripts/run_phase_a.py` against `plan_5_spatial_equity/tests/e2e/` and `PROJECT.md` M6 requirements to identify real white-box security/integrity gaps.

### 4.2 Real White-Box Gaps Discovered & Hardening Resolutions (发现的真实白盒漏洞与工程加固)

Four real edge-case gaps were discovered via white-box inspection, proved RED via tests first, and fixed cleanly:

1. **Page 2+ US Letter Geometry Blindspot in `audit_compiled_pdf()`**:
   - *Gap*: `pdfinfo` only outputs Page 1's `Page size:` summary line; an adversarial PDF with Page 1 = A4 and Page 2 = US Letter (`612 x 792 pts`) previously passed `a4_dimensions=True`.
   - *Fix*: Added per-page `pypdf` mediabox verification across every page in `audit_compiled_pdf()`, computing `non_a4_pages` and requiring `len(non_a4_pages) == 0`.
2. **Multi-Page Back-Matter Declaration Scanning Blindspot in `audit_compiled_pdf()`**:
   - *Gap*: Back-matter causal-word scanning only inspected pages literally containing `"Separate Contribution Statement"`, which would skip a subsequent page if `AI Usage Declaration` spilled onto a separate page.
   - *Fix*: Replaced single-page string check with a stateful `in_decl` flag scanning from the first declaration header through the final page of the PDF.
3. **Underscore Placeholder Blindspot (`TBD_NUM`) in `scan_report_text_issues()`**:
   - *Gap*: Regex `\bTBD\b` did not match `TBD_NUM` (because `_` is a regex word character).
   - *Fix*: Added case-sensitive `\bTBD_[A-Z0-9_]+\b` to `FORBIDDEN_PATTERNS` and the full-PDF artifact scan.
4. **Isolated Clone Root Override (`--project-root`) for Replay & Tamper Testing**:
   - *Gap*: `run_skip_download_pipeline()` and `scripts/run_phase_a.py` previously bound paths directly to `config.PROJECT_ROOT`, preventing isolated-clone E2E tamper and bit-identical replay tests without touching the real working tree.
   - *Fix*: Added optional `project_root: Path | None = None` to `run_skip_download_pipeline()` and `--project-root` to `scripts/run_phase_a.py`, enabling full isolated-clone Tier 4 & Tier 5 testing with zero mutation of the repository.

### 4.3 Implementation & TDD Workflow (代码实现与测试驱动开发)

Following `superpowers:test-driven-development`:
1. **RED Phase**: Built all 5 E2E tiers (`tests/e2e/test_tier1..tier5*.py`, 31 tests total) and ran `pytest tests/e2e -v` BEFORE applying the hardening patches. Confirmed all 8 Tier 5 single-fault clone mutations immediately caught their injected faults (`8/8` PASS), while all 4 white-box hardening targets (`--project-root`, `run_skip_download_pipeline(project_root=...)`, `TBD_NUM` + Page-2 US Letter in `audit_compiled_pdf()`, and missing M6 docs) failed RED cleanly as designed (plus aligned 4 exact committed JSON/CSV schema keys: `fitted_split`, `y_tail_pos`/`y_tail_rate`, 9 DM rows across `qlike`/`se`/`brier`, and `additivity_tolerance`).
2. **GREEN Phase**: Applied the 4 targeted hardening patches in `report.py`, `ingest/audit.py`, and `scripts/run_phase_a.py`, authored `plan_6_stock_volatility/TEST_READY.md` and this M6 milestone record, and updated all three READMEs/ledgers.
3. **FINAL VERIFICATION Phase**: Executed fresh `ruff check .` (`0` errors), `pytest tests/e2e --project-root . -v` (`31/31` passed), full `pytest -v` (`97/97` passed), `git check-ignore`, secret scan, blob size check (`< 25 MB`), and sibling/frozen-data isolation checks.

---

## 5. Verification Commands & Observed Results (独立验证命令与实测输出)

```bash
cd plan_6_stock_volatility
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync ruff check .
# Expected: All checks passed! (exit 0)

MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync pytest tests/e2e --project-root . -v
# Expected: 31 passed (Tier 1: 6, Tier 2: 7, Tier 3: 4, Tier 4: 5, Tier 5: 9; exit 0)

MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync pytest -v
# Expected: 97 passed (M0-M5: 66 + M6 E2E Tiers 1-5: 31; exit 0)

MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync python scripts/run_phase_a.py --skip-download
# Expected: Full Steps 01->06 JSON summary, "verdict": "PASS", 51/51 claims, git status clean (exit 0)

cd ..
git check-ignore -v --no-index plan_6_stock_volatility/docs/milestones/*.md plan_6_stock_volatility/TEST_READY.md
# Expected: empty stdout, exit code 1
```

---

## 6. Final Phase A Delivery & Git Hygiene Checklist (阶段 A 最终交付检查清单)

- [x] All 7 Plan 6 Phase A milestones (`M0` through `M6`) implemented, verified, documented in `docs/milestones/`, and pushed to GitHub `origin/main`
- [x] Full 6-step offline replay (`scripts/run_phase_a.py --skip-download`) verified bit-identical (`0` diffs)
- [x] Opaque-box E2E Tiers 1–4 (`22/22`) and White-box Tier 5 Adversarial Hardening (`9/9`, 8 single-fault clone mutations + auditor unit test) passing (`31/31`; `97/97` total repo tests)
- [x] `plan_6_stock_volatility/TEST_READY.md`, `docs/milestones/README.md`, `plan_6_stock_volatility/README.md`, and root `README.md` updated and git-tracked (`git check-ignore` exit `1`)
- [x] Zero sibling diffs in `plan_2_tourism_hotel/`, `plan_5_spatial_equity/`, `plans/`, and zero changes to frozen M1–M2 data (`plan_6_stock_volatility/data/`)
- [x] Zero `.agents/`, local coordination files (`/PROJECT.md`, `/TEST_INFRA.md`, `/TEST_READY.md`), LaTeX aux files, oversized blobs (`< 25 MB`), or secrets staged
