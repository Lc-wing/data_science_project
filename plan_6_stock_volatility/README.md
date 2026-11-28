# Corporate Event Shocks, Realized Volatility, and Downside Tail-Risk Forecasting in U.S. Equities (2019–2025)

Self-contained CS482 Data Science implementation workspace (`stock-volatility` / `src/stock_volatility`) for [Plan 6 (`plans/plan_6_stock_volatility_news_llm.md`)](../plans/plan_6_stock_volatility_news_llm.md), strictly adhering to the cross-project methodological guardrails in [`plans/00_review_report.md`](../plans/00_review_report.md).

> **Academic Research & Non-Investment-Advice Notice**
>
> This project is conducted solely for CS482 academic coursework and methodological evaluation of associational and predictive volatility and downside tail-risk models. It makes no causal claims and **does not constitute investment, financial, or trading advice**.
>
> 本项目仅用于 CS482 课程学术研究与方法学评估，不宣称因果机制，不构成任何投资或交易建议。

---

## 1. Report-Aligned Architecture Overview

Both the codebase (`src/stock_volatility/`) and sequential CLI pipeline (`scripts/01..07`) mirror the exact 5-section structure of the final CS482 submission report ([`report/CS482_GroupX_StockVolatilityLLM.pdf`](report/CS482_GroupX_StockVolatilityLLM.pdf), `4.46` A4 main-text pages, `61/61` programmatically audited numeric claims):

| Report Section | Package Modules & Scripts | Core Methodological Scope |
| --- | --- | --- |
| **1. Research Question** | `config.py`, `panel/spec.py` | Pre-registered universe (`60` liquid S&P 500 equities across `5` GICS sectors), non-causal scope boundaries, and falsifiable hypotheses **H1** (associational leverage & VIX), **H2** (tabular ML vs. classical econometrics), and **H3** (deep sequence & event-anchored LLM fusion; null results accepted). |
| **2. Data Collection & Preparation** | `ingest/*`, `panel/*`, `scripts/01_download_and_audit.py`, `scripts/02_build_panel_and_freeze.py` | Yahoo Finance daily OHLCV (`105,600` stock-days, `1,760` NYSE days), backward-only forward-filled FRED macro (`VIXCLS`, `T10Y2Y`), paginated SEC EDGAR Form `8-K`/`10-Q`/`10-K` filings (`7,703` unique filings; `BLK`/`XOM` predecessor CIKs), DST-aware `16:00 ET` (`13:00 ET` early-close) cutoffs, `<= 5`/day cap (`6,822` event-anchored days), `6`-day purged/embargoed chronological splits, and cryptographic SHA-256 freeze (`data/frozen/FREEZE_MANIFEST.json`). |
| **3. Data Analysis** | `analysis/eda.py`, `analysis/panel_har_rv.py`, `analysis/baselines.py`, `analysis/lgbm.py`, `analysis/treeshap.py`, **`analysis/dl/*`**, `scripts/03..06` | **(3.1) EDA & H1 Econometrics**: Cross-cycle volatility regimes & Stock-FE Extended Panel HAR-RV (`L=7` Bartlett Driscoll–Kraay SE, Holm family).<br>**(3.2) H2 Econometric & Tabular ML**: Quarterly expanding purged Persistence, per-stock GJR-GARCH(1,1) (`300/300` converged), Predictive/Logistic HAR-RV (Duan smearing), Global LightGBM, and Train-background exact TreeSHAP.<br>**(3.3) H3 Deep Time-Series & Event-Anchored LLMs (`analysis/dl/`)**: 5-seed multivariate `PatchTST` (`1M` vs. `15M`, non-crossing Pinball quantiles), zero-shot `Chronos-2` (`120M`) & `Chronos-T5-Large` (`710M`), `ProsusAI/finbert` (`110M`), and `google/gemma-4-12B-it` / `Qwen/Qwen3.8-27B` Zero-Shot & 4-bit NF4 QLoRA (`r=16, alpha=32` on text-only linear layers) fused onto LightGBM via 2024-Val out-of-fold meta-fusion. |
| **4. Method Evaluation** | `evaluation.py`, `analysis/lgbm.py`, `analysis/dl/fusion.py` | Variance-scale QLIKE & RMSE; imbalanced tail-risk PR-AUC (`8.62%` 2025 base rate), ROC-AUC, Brier, and 10-bin ECE; 4-quarter OOF probability calibration & Ridge/Logistic meta-fusion strictly inside `2024` Val; `5`-day date-block bootstrap 95% CIs (`B=1,000`) and `h=5` Bartlett-PSD HLN-modified Diebold–Mariano tests with Holm adjustment. |
| **5. Results & Discussion** | `figures.py`, `report.py`, `scripts/07_build_report.py` | Publication-grade 300-DPI Figure 1 & Figure 2 (PNG + vector PDF), unified Table 1 (H1 + H2) & Table 2 (H3 DL/LLM ablation), deterministic `SOURCE_DATE_EPOCH` LaTeX compilation (`report/CS482_GroupX_StockVolatilityLLM.pdf`), and automated 9-gate / 61-claim audit (`report/report_audit.md`). |

---

## 2. Key Empirical Summary (2019–2025 Panel & 2025 Single-Shot Holdout)

- **H1 (Stock-FE Extended Panel HAR-RV, $N=104{,}040$, Within $R^2 = 0.4736$)**: 5-sector average leverage (`-1.4660`, Holm $p < 0.0001$), lagged VIX (`+0.0156`, Holm $p < 0.0001$), and the joint 5-sector interaction Wald test ($\chi^2 = 43.39$, Holm $p < 0.0001$) are strongly significant across all sensitivities, while pairwise Technology vs. Healthcare leverage is non-significant under cross-sectionally robust Driscoll–Kraay SEs (Holm $p = 1.000$).
- **H2 (2025 Holdout Price-Only Forecasting, $N=14{,}700$, `245` Scorable Days)**: Global LightGBM achieves the lowest baseline volatility QLIKE (`0.2306`) and RMSE (`0.1306`), significantly beating Persistence (`0.2896`, Holm $p < 0.0001$) and GJR-GARCH (`0.2994`, Holm $p = 0.0001$). Frozen 2024-Val isotonic calibration cuts LightGBM 2025 ECE by `94.7%` (`0.1930 -> 0.0102`, Brier `0.0757`), whereas Logistic HAR-RV achieves higher PR-AUC overall (`0.1436` vs. `0.1231`) and on Form 8-K Item 2.02 earnings days (`0.2143` vs. `0.1190`).
- **H3 (2025 Holdout Deep Time-Series & Event-Anchored LLM Ablation, `895` Scorable Event Days)**: Scaling `PatchTST` from `1M` to `15M` improves QLIKE (`0.3142 -> 0.2819`) and nominal 80% coverage (`63.1% -> 68.3%`), and `chronos-2` (`0.3505`) improves over `chronos-t5-large` (`0.5848`), yet all four deep/foundation sequence models remain outperformed by tabular LightGBM (Holm $p \le 0.006$). Conversely, 2024-Val OOF text fusion significantly lowers 2025 full-universe QLIKE across all three text arms (`0.2251–0.2252`, HLN-DM Holm $p = 0.020$); on event days, compact domain-pretrained `FinBERT` (`110M`) lifts tail-risk PR-AUC from `0.1076` to `0.1398` (`+30.0%`), outperforming single-epoch 4-bit QLoRA on `google/gemma-4-12B-it` (`0.1108` zero-shot; `0.1076` QLoRA).

---

## 3. Directory Layout

```text
plan_6_stock_volatility/
├── .gitignore                             # Cache, raw archive, model weight, zip & LaTeX build ignores
├── README.md                              # Unified project guide aligned with the 5-section CS482 report
├── TEST_READY.md                          # 104-test verification & 5-tier E2E/adversarial audit guide
├── pyproject.toml                         # Self-contained uv + hatchling package, ruff, pytest config
├── uv.lock                                # Deterministic locked dependency graph
├── notebooks/
│   └── dl_gpu_training_colab.ipynb        # One-click Colab L4 (24GB) / A100 (40–80GB) GPU runner notebook
├── src/stock_volatility/
│   ├── __init__.py                        # Thread-determinism guards & package version
│   ├── config.py                          # Canonical paths, pre-registered constants, UNIVERSE_60, SHA-256 helpers
│   ├── ingest/                            # [Sec. 2] Universe, Yahoo OHLCV, FRED macro, SEC EDGAR, Day-1 audit
│   ├── panel/                             # [Sec. 2] 16:00/13:00 ET cutoff, <=t features, purged splits, freeze
│   ├── analysis/                          # [Sec. 3] Unified Data Analysis module (Econometrics + ML + DL/LLM)
│   │   ├── eda.py                         #   [Sec. 3.1] Volatility regimes, leverage, coverage, VIX x event
│   │   ├── panel_har_rv.py                #   [Sec. 3.1] H1 Stock-FE Panel HAR-RV + Bartlett Driscoll–Kraay SE
│   │   ├── baselines.py                   #   [Sec. 3.2] Purged folds, Persistence, GJR-GARCH(1,1), HAR-RV
│   │   ├── lgbm.py                        #   [Sec. 3.2] Global LightGBM + 2024-Val OOF calibration & H2 runner
│   │   ├── treeshap.py                    #   [Sec. 3.2] Train-background exact TreeSHAP & additivity audit
│   │   └── dl/                            #   [Sec. 3.3] Deep Time-Series & Event-Anchored Financial LLMs
│   │       ├── datasets.py                #     Frozen sequence windows (60d) & zero-leakage event prompts
│   │       ├── patchtst.py                #     Multivariate PatchTST (1M & 15M, ticker/sector emb, Pinball)
│   │       ├── chronos.py                 #     Zero-shot causal Chronos-2 (120M) & Chronos-T5-Large (710M)
│   │       ├── finbert.py                 #     Frozen ProsusAI/finbert (110M) pre-close event scoring
│   │       ├── qlora_llm.py               #     Gemma-4-12B-it / Qwen3.8-27B Zero-Shot & 4-bit NF4 QLoRA
│   │       └── fusion.py                  #     Temperature scaling, 2024-Val OOF meta-fusion & H3 Table 2
│   ├── evaluation.py                      # [Sec. 4] QLIKE/PR-AUC/ECE, block bootstrap & h=5 HLN-DM + Holm
│   ├── figures.py                         # [Sec. 5] 300-DPI Figure 1 & Figure 2 generators (PNG + PDF)
│   └── report.py                          # [Sec. 5] Table 1/2, unified CS482 LaTeX/PDF build & 61-claim audit
├── scripts/
│   ├── 01_download_and_audit.py           # Step 01: Raw data download or offline --skip-download Day-1 audit
│   ├── 02_build_panel_and_freeze.py       # Step 02: Panel alignment, purged splits & --check / --refreeze
│   ├── 03_eda.py                          # Step 03: Cross-cycle EDA & SEC coverage consistency tables
│   ├── 04_panel_har_rv.py                 # Step 04: H1 Stock-FE Extended Panel HAR-RV regression
│   ├── 05_forecast_backtest.py            # Step 05: H2 baselines, Global LightGBM, HLN-DM & TreeSHAP
│   ├── 06_dl_and_llm_gpu.py               # Step 06: Colab GPU PatchTST, Chronos, FinBERT & QLoRA LLM runner
│   ├── 07_build_report.py                 # Step 07: Build 300-DPI figures, Table 1/2, CS482 PDF & audit
│   └── run_pipeline.py                    # Unified offline CPU replay CLI (Steps 01 -> 05 -> 07)
├── data/
│   ├── raw/                               # Provenance-locked OHLCV, macro, SEC filings & data_provenance.json
│   ├── interim/                           # Aligned NYSE macro_daily.parquet & events_aligned.parquet
│   └── frozen/                            # Immutable SHA-256 bundle (panel_daily.parquet, FREEZE_MANIFEST.json)
├── outputs/
│   ├── figures/                           # 300-DPI Figure 1 & Figure 2 (.png + vector .pdf)
│   ├── metrics/                           # JSON metrics & audits (Day-1, EDA, H1, H2, SHAP, H3 DL/LLM)
│   ├── predictions/                       # All Val/Test .parquet predictions & event-day text/LLM scores
│   └── tables/                            # CSV, Markdown & LaTeX tables (EDA, H1, H2, SHAP, Table 1, Table 2)
├── report/
│   ├── CS482_GroupX_StockVolatilityLLM.pdf # Final unified 5-section CS482 submission report (4.46 main pp)
│   ├── report.tex                         # Programmatically generated LaTeX source
│   ├── references.bib                     # 18 verified scholarly BibTeX entries
│   └── report_audit.md                    # Automated 9-gate & 61/61 numeric claim traceability report
├── docs/
│   ├── milestones/                        # Historical M0–M9 milestone delivery & AI execution audit logs
│   └── superpowers/                       # Design specifications & implementation plans
└── tests/
    ├── conftest.py                        # Offline socket blocker, thread guards & --project-root fixture
    ├── test_smoke.py                      # Scaffold, universe, chronology, library & LaTeX/Poppler checks
    ├── test_ingest.py                     # Raw ingestion, predecessor CIKs, Day-1 audit & milestone docs
    ├── test_panel_and_freeze.py           # 16:00/13:00 ET cutoffs, <=5 cap, purged splits & SHA-256 freeze
    ├── test_eda.py                        # Cross-cycle EDA & non-causal wording contracts
    ├── test_regression.py                 # Driscoll–Kraay equivalence vs. statsmodels, Holm & H1 contracts
    ├── test_forecast.py                   # Purged expanding folds, Duan smearing, GJR-GARCH & calibrators
    ├── test_evaluation_and_shap.py        # QLIKE/ECE math, h=5 HLN-DM, block bootstrap & TreeSHAP additivity
    ├── test_figures_and_report.py         # 300-DPI figures, Table 1, unified CS482 PDF & replay CLI checks
    ├── test_dl_and_llm.py                 # PatchTST, Chronos, FinBERT, QLoRA filters, fusion & H3 contracts
    └── e2e/                               # 5-tier opaque-box E2E (Tiers 1–4) & Tier 5 single-fault suite
```

---

## 4. Environment Setup & Reproducible Execution

### 4.1 Local CPU Prerequisites

- **Python & `uv`**: Python `>= 3.11` and [`uv`](https://docs.astral.sh/uv/).
- **TeX Live & Poppler** (required to compile and audit `report/CS482_GroupX_StockVolatilityLLM.pdf`): `pdflatex`, `bibtex`, `pdfinfo`, and `pdftotext`. On Ubuntu/Debian:

```bash
sudo DEBIAN_FRONTEND=noninteractive apt-get update && \
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  texlive-latex-base texlive-latex-recommended texlive-latex-extra \
  texlive-fonts-recommended texlive-plain-generic lmodern \
  poppler-utils poppler-data
```

Install the locked virtual environment inside `plan_6_stock_volatility/`:

```bash
cd plan_6_stock_volatility
uv sync --all-groups
```

### 4.2 Local Offline CPU Replay & Report Rebuild

All tabular, econometric, TreeSHAP, figure, and final CS482 PDF compilation steps run deterministically offline on CPU (reusing the committed Colab L4 H3 artifacts in `outputs/predictions/`, `outputs/metrics/`, and `outputs/tables/`):

```bash
# Fast report-only rebuild (regenerates 300-DPI Figure 1 & 2, Table 1, PDF & 61-claim audit):
uv run python scripts/run_pipeline.py --report-only

# Complete offline CPU pipeline replay (Steps 01 -> 02 -> 03 -> 04 -> 05 -> 07; zero network calls):
uv run python scripts/run_pipeline.py --skip-download

# Individual step CLIs:
uv run python scripts/01_download_and_audit.py --skip-download
uv run python scripts/02_build_panel_and_freeze.py --check
uv run python scripts/03_eda.py
uv run python scripts/04_panel_har_rv.py
uv run python scripts/05_forecast_backtest.py
uv run python scripts/07_build_report.py
```

### 4.3 Colab Single-GPU Deep Learning & QLoRA LLM Execution (Step 06)

`scripts/06_dl_and_llm_gpu.py` enforces two safety gates (`--allow-weight-download` and `torch.cuda.is_available()`) so multi-gigabyte Hugging Face checkpoints are never downloaded accidentally on CPU machines. To reproduce or extend the H3 deep learning & LLM experiments on Google Colab (**NVIDIA L4 24GB** or **A100 40–80GB**), open [`notebooks/dl_gpu_training_colab.ipynb`](notebooks/dl_gpu_training_colab.ipynb) or run:

```bash
# Automatic VRAM-aware LLM selection:
#   - < 35 GB VRAM (Colab L4 24GB)  -> google/gemma-4-12B-it (4-bit NF4 QLoRA, ~15.3 GB peak VRAM)
#   - >= 35 GB VRAM (Colab A100)    -> Qwen/Qwen3.8-27B      (4-bit NF4 QLoRA)
python scripts/06_dl_and_llm_gpu.py \
  --allow-weight-download \
  --resume \
  --llm-model auto \
  --llm-epochs 1 \
  --routes all
```

Supported subset stages via `--routes`: `patchtst`, `chronos`, `finbert`, `llm`, or `all` (unselected stages automatically hydrate saved predictions and metrics from `outputs/`). After copying updated GPU outputs back to `outputs/`, run `uv run python scripts/07_build_report.py` locally to recompile and audit the final PDF.

### 4.4 Linting & 104-Test Verification Suite

```bash
uv run ruff check .
uv run pytest -v
uv run pytest tests/e2e --project-root . -v
```

See [`TEST_READY.md`](TEST_READY.md) for the complete 5-tier E2E and single-fault adversarial test matrix.

---

## 5. Anti-Leakage & Reproducibility Protocol

1. **Zero Survivorship Lookahead**: Fixed pre-2019 universe of `60` liquid S&P 500 constituents (`12` per GICS sector; Consumer balanced `6` Staples + `6` Discretionary under pre-2019 GICS taxonomy).
2. **DST & Early-Close SEC Timestamp Alignment**: Second-precision UTC `acceptanceDateTime` stamps are converted to `America/New_York` (`16:00 ET` regular close; `13:00 ET` on all `15` NYSE early-close half-days). Post-close and non-trading-day filings roll strictly to the next NYSE trading day `t+1`.
3. **Train-Only Preprocessing & Thresholds**: Continuous sequence Winsorization (`[0.5%, 99.5%]`), mean/std standardization, VIX regime median (`19.99`), `rv90` sensitivity threshold (`0.3694`), and `1,000`-row TreeSHAP background are fit exclusively on `2019–2023` Train (`<= 2023-12-20`).
4. **Purged & Embargoed Chronological Splits**: `6`-trading-day embargoes (`>= h=5`) separate Train (`2019-01-02..2023-12-20`), Val (`2024-01-02..2024-12-20`), and single-shot Holdout Test (`2025-01-02..2025-12-31`, final `5` incomplete forward-target days masked `NA`), with `>= 5`-day purges before every quarterly expanding refit (`2025Q1`–`2025Q4`).
5. **2024-Val Out-of-Fold Calibration & Meta-Fusion**: Both probability calibration (Isotonic vs. Platt vs. Temperature Scaling) and H3 multimodal Ridge/Logistic meta-fusion are selected via 4-quarter OOF cross-validation strictly inside `2024` Val so in-sample QLoRA memorization cannot distort fusion weights.

---

## 6. Milestone Delivery History

Complete chronological engineering, verification, and AI execution audit records are maintained in [`docs/milestones/README.md`](docs/milestones/README.md):

- [`docs/milestones/M0_repo_cleanup_and_scaffold.md`](docs/milestones/M0_repo_cleanup_and_scaffold.md) — Toolchain, universe & scaffold (`124fc67`)
- [`docs/milestones/M1_data_ingest_and_day1_audit.md`](docs/milestones/M1_data_ingest_and_day1_audit.md) — OHLCV, FRED macro & paginated SEC EDGAR ingestion (`5370d18`)
- [`docs/milestones/M2_panel_splits_and_freeze.md`](docs/milestones/M2_panel_splits_and_freeze.md) — 16:00/13:00 ET cutoffs, purged splits & SHA-256 freeze (`3c9bba9`)
- [`docs/milestones/M3_eda_and_h1_panel_har_rv.md`](docs/milestones/M3_eda_and_h1_panel_har_rv.md) — Cross-cycle EDA & H1 Stock-FE Panel HAR-RV (`d7a35b6`)
- [`docs/milestones/M4_h2_forecasting_calibration_and_shap.md`](docs/milestones/M4_h2_forecasting_calibration_and_shap.md) — H2 baselines, LightGBM, HLN-DM & TreeSHAP (`744df5d`)
- [`docs/milestones/M5_figures_table1_and_cs482_report.md`](docs/milestones/M5_figures_table1_and_cs482_report.md) — 300-DPI figures, Table 1 & deterministic LaTeX builder (`63dc8ec`)
- [`docs/milestones/M6_e2e_and_adversarial_hardening.md`](docs/milestones/M6_e2e_and_adversarial_hardening.md) — 5-tier E2E & white-box adversarial suite (`547b286`)
- [`docs/milestones/M7_phase_b_colab_and_dl_pipeline.md`](docs/milestones/M7_phase_b_colab_and_dl_pipeline.md) — Deep sequence & QLoRA LLM GPU pipeline (`1b6827f`)
- [`docs/milestones/M8_phase_b_l4_results_and_final_pdf.md`](docs/milestones/M8_phase_b_l4_results_and_final_pdf.md) — Colab NVIDIA L4 execution & 5-section CS482 report (`c1a2139`)
- [`docs/milestones/M9_unified_repository_layout.md`](docs/milestones/M9_unified_repository_layout.md) — Report-aligned package/outputs/CLI unification & documentation rewrite
