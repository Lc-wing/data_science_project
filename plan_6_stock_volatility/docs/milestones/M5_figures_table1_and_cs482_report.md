# Milestone M5: 300-DPI Figures, Unified Table 1, Audited CS482 Phase A Draft PDF & Single-Command Replay CLI (里程碑 M5：300-DPI 双图、统一主表 Table 1、CS482 阶段 A 报告 PDF 自动化审计与一键离线复现 CLI)

- **Milestone**: `M5 — 300-DPI Figures, Table 1, CS482 Phase A Draft PDF & Replay CLI (300-DPI 双图、Table 1、CS482 阶段 A 报告 PDF 与一键复现 CLI)`
- **Status**: `COMPLETED & VERIFIED`
- **Parent / Baseline Commit SHA**: `744df5dd8647aaae93e5f4b1049273c781b9bb92` (`M4` on `origin/main`)
- **Date (UTC)**: `2026-11-23`
- **Verification Verdict**: `VERIFIED PASS` (`ruff check .`: 0 errors; `pytest -v`: `66/66` passed incl. 6 M5 TDD/figure/PDF/CLI/doc tests; `report/phase_a_audit.md`: `PASS` with `9/9` CS482 gates, `51/51` numeric claims traced & verified, `0` forbidden causal words / `??` / `NaN`, exact A4 `595.28 x 841.89 pts`, `4.00` A4 main-text pages; `run_phase_a.py --report-only`: exit `0`; `git check-ignore`: `1`; `0` sibling diffs; `0` secrets)

---

## 1. Milestone Objectives & Scope (里程碑目标与范围)

Milestone M5 converts the frozen M1–M4 empirical artifacts into publication-grade CS482 deliverables with 100% programmatic claim traceability and single-command offline reproducibility:

1. **300-DPI Publication Figures (`src/stock_volatility/figures.py`)**:
   - **Figure 1 (`fig1_volatility_regimes_and_forecasts.{png,pdf}`)**: 3-panel Okabe–Ito colorblind-safe layout:
     - *Panel A*: Cross-sector 20-day smoothed 5-day annualized Parkinson realized volatility across 2019–2025 with shaded COVID-19 (2020), Rate-Hike Bear (2022), 2024 Validation, and 2025 Holdout regimes.
     - *Panel B*: Lagged daily return bucket (`r_{i,t}`) vs. mean forward 5-day RV leverage asymmetry across All 60 Stocks, Technology, Financials, and Healthcare.
     - *Panel C*: Single-shot 2025 holdout forecast tracking during the March–May 2025 volatility spike comparing Actual RV against Global LightGBM, Panel HAR-RV, Per-Stock GJR-GARCH(1,1), and Persistence.
   - **Figure 2 (`fig2_shap_and_calibration.{png,pdf}`)**: 2-panel explainability & probability calibration figure:
     - *Panel A*: Train-background (`N=1,000`) exact TreeSHAP attributions for LightGBM tail-risk model behavior split by High-VIX (`>= 19.61`) vs. Low-VIX (`< 19.61`) regimes (strictly non-causal, positive class 1 verified, additivity error `4.1e-08`).
     - *Panel B*: 10-bin 2025 holdout calibration reliability curves comparing LightGBM Raw (`ECE=0.380`), LightGBM 2024-Val OOF Isotonic (`ECE=0.010`, `-97.3%`), Logistic HAR-RV (`ECE=0.007`), and GJR-GARCH (`ECE=0.026`).
2. **Unified Main Summary Table (`outputs/tables/table1_summary.{csv,md,tex}`)**:
   - Combines **Panel A (H1 Stock-FE Extended Panel HAR-RV, `N=104,040`, Bartlett Driscoll–Kraay SE `L=7`, 5-test Holm confirmatory family)** and **Panel B (H2 Single-Shot 2025 Holdout Test, `N=14,700`, 5-day date-block bootstrap `B=1,000` 95% CIs, and `h=5` HLN-DM Holm-adjusted `p`-values vs. LightGBM)** directly from committed M3/M4 JSON/CSV files.
3. **Programmatic CS482 Phase A Draft PDF & Automated Audit (`src/stock_volatility/report.py`, `scripts/06_write_report.py`)**:
   - Programmatically collects **51 numeric claims** directly from committed M1–M4 artifacts (zero hardcoded copy-paste), writes verified BibTeX (`report/references.bib`, 13 entries including `kassis2026skills` `arXiv:2609.00065`), writes `report/phase_a_draft.tex`, compiles via `pdflatex -> bibtex -> pdflatex x2` into `report/CS482_GroupX_StockVolatilityLLM_PhaseA_Draft.pdf`, and generates `report/phase_a_audit.md` auditing A4 geometry, CS482 `[3.0, 5.0]` A4 main-text page budget (cover, references, contribution statement, and AI declaration excluded), all 51 numeric claims, non-investment-advice notice, and zero forbidden causal words (`impact`, `driver`, `elasticity`, `caused`, `causes`) or LaTeX artifacts (`??`, `TODO`, `TBD`, `NaN`).
4. **One-Command Offline Phase A Replay CLI (`scripts/run_phase_a.py`)**:
   - Orchestrates Steps 01–06 end-to-end with `--skip-download` (default offline replay from committed snapshots), `--report-only` (fast M5 figure/Table 1/PDF/audit rebuild), and `--refreeze` (optional deterministic M2 refreeze).
5. **Write-Ownership Isolation**:
   - Strictly touches only M5 code, scripts, tests, figures, Table 1, report files, and milestone documentation. Zero edits to `plan_2_tourism_hotel/`, `plan_5_spatial_equity/`, `plans/`, or frozen M1–M2 data files (`data/`).

---

## 2. Milestone Deliverables & File Inventory (交付物清单与文件契约)

| File Path (Relative to `plan_6_stock_volatility/`) | Category | Size | SHA-256 / Contract Summary |
| --- | --- | ---: | --- |
| `src/stock_volatility/figures.py` | Code | 16,147 B | 300-DPI `generate_figure1()`, `generate_figure2()`, `generate_all_figures()` (deterministic PDF `CreationDate=None`) |
| `src/stock_volatility/report.py` | Code | 55,602 B | `build_table1_summary()`, `collect_report_claims()` (51 claims), `render_references_bib()`, `render_phase_a_tex()`, deterministic `SOURCE_DATE_EPOCH` `compile_latex_pdf()`, `audit_compiled_pdf()`, `run_report_build_and_audit()` |
| `scripts/06_write_report.py` | CLI | 473 B | Offline M5 build & automated audit entrypoint (`verify_freeze_manifest()` enforced first) |
| `scripts/run_phase_a.py` | CLI | 3,080 B | One-command Phase A replay CLI (`--skip-download`, `--report-only`, `--refreeze`) |
| `tests/test_figures_and_report.py` | Test | 8,545 B | 6 M5 TDD tests (Table 1 builder, causal/artifact scanner, 300-DPI PNG/PDF metadata, A4 PDF + 51-claim audit, `--report-only` CLI, M5 milestone doc & gitignore check) |
| `outputs/figures/fig1_volatility_regimes_and_forecasts.png` | Output | 764,463 B | `964ce2e6b34c1854adc62caed78be8df687d3285f49fe4c1d5c331733801c7db` (300 DPI PNG, pHYs verified) |
| `outputs/figures/fig1_volatility_regimes_and_forecasts.pdf` | Output | 89,808 B | `5d5fe3e18edf7cc75c29664cca10c7105bc56fcc96999b235b8b7e1c1db3836c` (Bit-identical vector PDF embedded in LaTeX) |
| `outputs/figures/fig2_shap_and_calibration.png` | Output | 360,590 B | `5c2ad1a24a8688e455545b36dd08c2a2a44ed4447c79476b508cd35c2c19c63d` (300 DPI PNG, pHYs verified) |
| `outputs/figures/fig2_shap_and_calibration.pdf` | Output | 41,457 B | `a02aa4e93172b5bd8646c1377ac0bca3f20cd879479bc02d42f11d1f68a65072` (Bit-identical vector PDF embedded in LaTeX) |
| `outputs/tables/table1_summary.csv` | Output | 1,146 B | `34f6330be800bacbd8ed1e2529473478bbaef8e8e80ef8b87e67e68772e29d2f` (9 rows: 5 H1 + 4 H2) |
| `outputs/tables/table1_summary.md` | Output | 1,515 B | `dec2edbe951ee10abe0d2877c2cb0f07f813ae4378666b7242f9350451ce8f37` (Markdown Panel A + Panel B) |
| `outputs/tables/table1_summary.tex` | Output | 1,969 B | `704e3301d11090b051302249665995df1ec404f5daa691d7d2c84e6ed9283245` (Booktabs LaTeX table) |
| `report/references.bib` | Report | 3,919 B | `c6b9d84e6700bf75b8db330e983b6f221ee62fe2eec167c1941e4ded3baded26` (13 verified BibTeX entries) |
| `report/phase_a_draft.tex` | Report | 19,592 B | `ab1d5133cae45b07c32ce21160a3f6ed8131f21d1b17316ec120cef25917c903` (Single-column A4 11pt 1.5-spacing LaTeX source) |
| `report/CS482_GroupX_StockVolatilityLLM_PhaseA_Draft.pdf` | Report | 429,991 B | `879c5e6a7b8ce2a429ded418bc52a60c55346f88b378a20e2d8510eec3eb6e4b` (Bit-identical 7-page A4 PDF: 1 Cover + 4.00 Main + 1 Refs + 1 Declarations) |
| `report/phase_a_audit.md` | Report | 6,477 B | `f7ede5a3743d4865716c6189103e602c368767eb7a6ec71033387f8b82381362` (`PASS`, 9/9 gates, 51/51 claims, 0 wording/artifact hits) |
| `docs/milestones/M5_figures_table1_and_cs482_report.md` | Doc | — | This M5 delivery & AI execution record |
| `docs/milestones/README.md`, `README.md`, `../README.md` | Doc | — | Updated milestone ledger with M4 SHA `744df5dd8647aaae93e5f4b1049273c781b9bb92` and M5 completion status |

---

## 3. Technical Implementation & Key Audit Metrics (核心技术实现与关键审计指标)

### 3.1 CS482 Formatting & Automated Audit Gates (`report/phase_a_audit.md`)

| Audit Dimension | Observed Result | CS482 / Plan 6 Requirement | Gate Status |
| --- | --- | --- | --- |
| Overall Audit Verdict | `PASS` | Must pass all 9 checks | `PASS` |
| Page Geometry (`pdfinfo`) | `595.28 x 841.89 pts` | A4 (`595.28 x 841.89 pts`) | `PASS` |
| Layout & Typography | Single-column, `11pt`, `\onehalfspacing` (1.5 spacing) | Single-column, 11pt, 1.5 spacing | `PASS` |
| Page Structure | `7` total pages (`1` Cover + `4` Main + `1` References + `1` Contribution & AI Declaration) | Cover, References, Separate Contribution Statement & AI Declaration excluded | `PASS` |
| Main Text Length | `4.00` A4 pages (`4` occupied, `0` orphan lines) | `[3.00, 5.00]` A4 pages (leaves `1.00` full A4 page for Phase B) | `PASS` |
| Numeric Claim Traceability | `51 / 51` claims programmatically injected & verified in extracted PDF text | Zero manual copy-paste / zero drift from frozen JSON/CSV | `PASS` |
| Forbidden Causal Words & Artifacts | `0` hits (`impact`, `driver`, `elasticity`, `caused`, `causes`, `??`, `TODO`, `TBD`, `NaN`) | Zero causal wording outside BibTeX titles; zero unresolved LaTeX refs | `PASS` |
| Non-Investment-Advice & Academic Scope | Present on Cover Page, Section 1, and Section 5 | Mandatory academic research & non-investment disclaimer | `PASS` |
| K-Dense Scientific Skills Citation | `Kassis et al. (2026)`, `arXiv:2609.00065` cited & in `references.bib` | Mandatory Plan 6 / `00_review_report.md` acknowledgement | `PASS` |

### 3.2 Key Programmatic Empirical Results Verified in Table 1 & Draft PDF

- **Freeze Provenance**: `data/frozen/panel_daily.parquet` SHA-256 prefix `3515fcd12cab` verified via `verify_freeze_manifest()` before any figure or report build; `105,600` stock-days (`60` stocks × `1,760` NYSE days), `7,703` raw paginated SEC EDGAR filings (`195` archives), `6,822` event-anchored stock-days frozen for Phase B (`2` early-close `WFC` filings shifted; `29` duplicates removed).
- **H1 Confirmatory Panel HAR-RV (`N=104,040`, Within `R²=0.4736`, Bartlett DK `L=7`)**: Overall leverage AME `-1.4458` (`p_holm < 0.0001`), VIX AME `+0.0141` (`p_holm < 0.0001`), joint 5-sector interaction Wald `χ²=84.84` (`F=10.60`, `p_holm < 0.0001`); pairwise Technology vs. Healthcare interactions non-significant under conservative DK SEs + Holm (`p_holm = 0.171` / `0.380`).
- **H2 Single-Shot 2025 Holdout (`N=14,700`, `245` scorable days, `300/300` GJR-GARCH converged)**:
  - **Regression**: Global LightGBM lowest QLIKE (`0.2306`) & RMSE (`0.0951`), beating Persistence (`0.3144`, Holm `p < 0.0001`) and GJR-GARCH (`0.3007`, Holm `p = 0.0207`), with largest sector gain in Technology (`0.2862` vs. Panel HAR-RV `0.3369`; universe vs. HAR-RV `0.2383`, Holm `p = 0.134`).
  - **Classification & Calibration**: Frozen 2024-Val OOF `isotonic` calibrator cuts LightGBM 2025 ECE by `97.3%` (`0.3804 -> 0.0102`) and achieves lowest Brier (`0.0773`, Holm `p = 0.0470` vs. Persistence); Logistic HAR-RV achieves highest PR-AUC (`0.1422` vs. LightGBM `0.1231`; earnings days `0.2861` vs. `0.0948`).
  - **Train-Background TreeSHAP (`bg=1,000`)**: Max additivity error `4.07e-8` (clf) / `8.35e-8` (reg) `< 1e-6`; top classifier attributions are `rv_m` (`24.4%`), `vix_close` (`11.9%`), and `t10y2y`, amplified under High-VIX (`>= 19.61`).

---

## 4. What the AI Agents Did in This Milestone (本里程碑内 AI 完成的工作全记录)

### 4.1 Exploration & Specification Alignment (前期调研与规范对齐)

- Inspected CS482 project rubric (`Project-requirements.md`: 3–5 A4 pages main text, single-column, 11pt, 1.5 spacing; Cover Page with member names/IDs, References, Separate Contribution Statement, AI Usage Declaration excluded), Plan 6 v2.2 (`plans/plan_6_stock_volatility_news_llm.md`), review guardrails (`plans/00_review_report.md` C1–C15), and `plan_2_tourism_hotel/report/phase_a_audit.md` as structural reference.
- Verified exact schemas and JSON keys of all committed M1–M4 artifacts (`day1_data_audit.json`, `split_metadata.json`, `feature_manifest.json`, `FREEZE_MANIFEST.json`, `eda_summary.json`, `h1_regression.json`, `h2_baselines.json`, `h2_lgbm_selection.json`, `h2_evaluation.json`, `shap_audit.json`, `h2_test_metrics.csv`, `h2_dm_tests.csv`, `h2_test_by_sector.csv`, `h2_test_by_event_day.csv`) so every reported number is programmatically extracted.

### 4.2 Traps Discovered & Engineering Resolutions (发现的陷阱与工程解决)

Every bug, visual flaw, or reproducibility gap discovered during build, lint, test, and pixel-level visual QA was systematically resolved:

1. **JSON Audit Key Mismatch (`overlapping_archives_fetched`)**: Initial draft of `collect_report_claims()` queried `day1["sec_edgar"]["archives_fetched"]` and raised `KeyError`; corrected to the exact M1 schema key `overlapping_archives_fetched` (`195`).
2. **Python f-string Brace Escaping in LaTeX**: `$B=1{,}000$` inside `rf"""..."""` failed parsing and `rf"...}}"` dropped one closing brace in `\multicolumn{7}{l}{\textbf{...}}`; fixed via double-brace escaping (`$B=1{{,}}000$` and `}}}}`).
3. **Python Non-Raw BibTeX Accent Bug (`L'opez` vs. `López`)**: `render_references_bib()` initially used `"""` instead of `r"""`, turning `L{\'o}pez` into `L{'o}pez` and rendering `L'opez de Prado` in the PDF! Visual inspection of rendered page PNGs caught this immediately; switching to `r"""` restored proper `Marcos López de Prado`.
4. **LaTeX `--skip-download` En-Dash Ligature Inside `\texttt`**: Visual QA of page 7 caught `\texttt{... --skip-download}` rendering `--` as a single en-dash (`-skip-download`); fixed via `-{}-skip-download`.
5. **Cover Page Overfull `\hbox`, `hyperref` Duplicate `page.1`, and `pypdf` Tight-Kerning Extraction**: Added `\pagenumbering{roman}` / `\pagenumbering{arabic}` around `titlepage`, formatted the Cover Page member table at `\small`, and added `\raggedright` to the Cover Page notice minipage (preventing `pypdf` from collapsing word spaces on tight justified lines like `notinvestmentadvice`).
6. **Figure 1 & Figure 2 Visual QA Fixes (Pixel-Level Inspection of 300-DPI PNGs)**:
   - *Figure 1*: Panel A legend overlapped regime annotations and Panel B/C titles collided -> moved Panel A legend cleanly above the axes (`bbox_to_anchor=(0.5, 1.02)`), increased `hspace`/`wspace`, and shortened Panel B/C subtitles.
   - *Figure 2*: Panel A/B titles collided, bottom y-tick displayed raw `rv_d`, and Panel B calibration callout box overlapped the orange raw curve -> added explicit human-readable label `Daily RV (rv_d)`, widened subplot spacing, and repositioned the callout box cleanly into empty whitespace.
7. **LaTeX Float-Only Page & 4-Word Orphan Line on Page 5**: Initial build put Table 1 + Figure 2 on a sparse float-only page 4 and pushed 14 lines (then 4 words) onto page 5. Tuning LaTeX float fractions (`\topfraction=0.90`, `\textfraction=0.08`, `\floatpagefraction=0.82`), setting `\begin{table}[!t]`, `\begin{figure}[!tb]`, `\begin{figure}[!b]` (`width=0.84\textwidth`), and adding `\clearpage` before `Separate Contribution Statement` produced an exact, balanced **7-page PDF** with **`4.00` A4 main-text pages** (0 orphan lines) and References / Declarations cleanly separated.
8. **Bit-Identical PDF Determinism (`CreationDate=None` + `SOURCE_DATE_EPOCH`)**: Matplotlib PDFs and `pdflatex` originally embedded wall-clock `/CreationDate` timestamps, causing spurious SHA-256 drift across repeated offline replays. Passing `metadata={"CreationDate": None}` in `figures.py` and `SOURCE_DATE_EPOCH` + `FORCE_SOURCE_DATE=1` + `TZ=UTC` (anchored to `FREEZE_MANIFEST.json`'s `generated_at_utc`) in `compile_latex_pdf()` achieved **100% bit-identical SHA-256 parity across all 11 output & report files** on consecutive runs (`diff -u` empty).
9. **Audit Markdown Parser Contract, `ruff` E501 Compliance & Replay CLI Step 05 Loader**: Cleaned up 36 E501 warnings in `report.py`, removed a dead ternary branch, aligned audit markdown labels (`Overall Audit Verdict: **PASS**`, `Main Text Length: \`4.00\` A4 pages`), and replaced an initial placeholder import in `scripts/run_phase_a.py` with `importlib.util.spec_from_file_location` loading sibling `scripts/05_forecast_backtest.py`.

### 4.3 Implementation & TDD Workflow (代码实现与测试驱动开发)

Following `superpowers:test-driven-development`:
1. **RED Phase**: Authored `tests/test_figures_and_report.py` (6 tests covering `build_table1_summary`, `scan_report_text_issues`, 300-DPI PNG `pHYs` metadata + vector PDFs, A4 `pdfinfo` + `pypdf` required phrases + `phase_a_audit.md` PASS + `[3.0, 5.0]` page budget, offline `run_phase_a.py --report-only`, and M5 milestone delivery doc + `git check-ignore`) BEFORE implementing `figures.py`, `report.py`, or `run_phase_a.py`. Verified initial run failed `6/6` (`6 failed in 0.62s`).
2. **GREEN Phase**: Implemented `figures.py`, `report.py`, `scripts/06_write_report.py`, and `scripts/run_phase_a.py`, generated all M5 figures/Table 1/PDF/audit artifacts, visually inspected both 300-DPI figures and all 7 rendered PDF pages via `pdftoppm`, and iterated until all layout and visual defects were eliminated.
3. **REFACTOR & VERIFY Phase**: Wrapped all source lines for `ruff check .` (`0` errors), deleted temporary `/tmp/plan6_pdf_preview*` folders, wrote this M5 milestone delivery record, updated all three READMEs/ledgers, and ran full fresh `ruff`, `pytest -v` (`66/66`), and `run_phase_a.py --report-only` verification gates.

---

## 5. Verification Commands & Observed Results (独立验证命令与实测输出)

```bash
cd plan_6_stock_volatility
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync ruff check .
# Expected: All checks passed! (exit 0)

MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync pytest -v
# Expected: 66 passed (26 M0 + 9 M1 + 8 M2 + 8 M3 + 9 M4 + 6 M5; exit 0)

MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync python scripts/run_phase_a.py --report-only
# Expected: JSON summary with "verdict": "PASS", "main_pages_occupied": 4, "main_text_pages_est": 4.0, "claims_verified": 51/51 (exit 0)

cd ..
git check-ignore -v --no-index plan_6_stock_volatility/docs/milestones/*.md
# Expected: empty stdout, exit code 1
```

---

## 6. Git Hygiene & Delivery Checklist (Git 交付检查清单)

- [x] Dual-content record complete (M5 deliverables + exact SHA-256 checksums + full AI exploration/visual QA/TDD log)
- [x] `docs/milestones/README.md`, `plan_6_stock_volatility/README.md`, and root `README.md` updated with M4 commit SHA `744df5dd8647aaae93e5f4b1049273c781b9bb92` and M5 links
- [x] `git check-ignore -v --no-index plan_6_stock_volatility/docs/milestones/*.md` exited `1` (not gitignored)
- [x] Zero sibling diffs in `plan_2_tourism_hotel/`, `plan_5_spatial_equity/`, `plans/`, and frozen `plan_6_stock_volatility/data/`
- [x] Zero LaTeX auxiliary files (`.aux`, `.bbl`, `.blg`, `.log`, `.out`) or `.agents/` files staged; all staged files `< 25 MB`; zero secrets/tokens
