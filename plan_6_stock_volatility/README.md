# Plan 6: US Equity Realized Volatility & Downside Tail-Risk Forecasting (Phase A)

Implementation workspace for [Plan 6 (`plans/plan_6_stock_volatility_news_llm.md`)](../plans/plan_6_stock_volatility_news_llm.md), adhering to the cross-plan review guardrails in [`plans/00_review_report.md`](../plans/00_review_report.md).

This folder is a self-contained `uv` project (`stock-volatility` / `src/stock_volatility`) strictly isolated from sibling workspaces (`plan_2_tourism_hotel/` and `plan_5_spatial_equity/`).

> **Academic Research & Non-Investment-Advice Notice**
>
> This project is conducted solely for CS482 academic coursework and methodological evaluation of associational and predictive volatility/tail-risk models. It makes no causal claims and **does not constitute investment, financial, or trading advice**.
>
> 本项目仅用于 CS482 课程学术研究与方法学评估，不宣称因果机制，不构成任何投资或交易建议。

## Phase A Progress & Milestones

Phase A is strictly CPU-reproducible and freezes the aligned panel, splits, and corporate-event text dataset for downstream Phase B (single-A100 27B–64B QLoRA, out of scope for Phase A).

| Milestone | Scope | Status | Delivery & AI Execution Doc |
| --- | --- | --- | --- |
| M0 | Repository hygiene, TeX Live + Poppler toolchain, isolated `uv` scaffold, `config.py`, smoke tests | done (`124fc67`) | [`docs/milestones/M0_repo_cleanup_and_scaffold.md`](docs/milestones/M0_repo_cleanup_and_scaffold.md) |
| M1 | 60-stock OHLCV, macro (`^VIX`, FRED `VIXCLS`/`T10Y2Y`, `SPY`), paginated SEC EDGAR `8-K/10-Q/10-K`, Day-1 & coverage audit, milestone delivery docs | done (`5370d18`) | [`docs/milestones/M1_data_ingest_and_day1_audit.md`](docs/milestones/M1_data_ingest_and_day1_audit.md) |
| M2 | Strict `16:00 America/New_York` (`13:00 ET` early-close) cutoff, deduplication & `<= 5`/day cap, features/targets, Purged+Embargoed splits, SHA-256 freeze | done (`3c9bba9`) | [`docs/milestones/M2_panel_splits_and_freeze.md`](docs/milestones/M2_panel_splits_and_freeze.md) |
| M3 | Cross-cycle EDA & H1 Stock-FE Extended Panel HAR-RV with Driscoll–Kraay SE & Holm-adjusted CIs | done (`d7a35b6`) | [`docs/milestones/M3_eda_and_h1_panel_har_rv.md`](docs/milestones/M3_eda_and_h1_panel_har_rv.md) |
| M4 | H2 Out-of-sample baselines (Persistence, GJR-GARCH, HAR-RV), Global LightGBM, 2024 Val calibration, 2025 Holdout Test, `h=5` HLN-DM/Holm & Train-background TreeSHAP | done (`744df5d`) | [`docs/milestones/M4_h2_forecasting_calibration_and_shap.md`](docs/milestones/M4_h2_forecasting_calibration_and_shap.md) |
| M5 | 300-DPI Figure 1 & Figure 2, Table 1, audited CS482 Phase A Draft PDF (`4.00` A4 main pp), and one-command replay CLI | done (`63dc8ec`) | [`docs/milestones/M5_figures_table1_and_cs482_report.md`](docs/milestones/M5_figures_table1_and_cs482_report.md) |
| M6 | Opaque-box E2E Tiers 1–4 verification, Tier 5 white-box adversarial hardening & final Phase A sign-off | done | [`docs/milestones/M6_e2e_and_adversarial_hardening.md`](docs/milestones/M6_e2e_and_adversarial_hardening.md) / [`TEST_READY.md`](TEST_READY.md) |

See [`docs/milestones/README.md`](docs/milestones/README.md) for the master milestone delivery ledger, the mandatory per-milestone AI execution recording policy, and the standard template for Milestones M0–M6.

## Prerequisites & Setup

- **Python & `uv`**: Python `>= 3.11` and [`uv`](https://docs.astral.sh/uv/).
- **LaTeX & Poppler** (required for CS482 PDF build and numeric/layout audits): `pdflatex`, `bibtex`, and Poppler (`pdfinfo`, `pdftotext`). On Ubuntu/Debian:

```bash
sudo DEBIAN_FRONTEND=noninteractive apt-get update && \
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  texlive-latex-base \
  texlive-latex-recommended \
  texlive-latex-extra \
  texlive-fonts-recommended \
  texlive-plain-generic \
  lmodern \
  poppler-utils \
  poppler-data
```

Sync the isolated environment and run tests and lint checks from inside `plan_6_stock_volatility/`:

```bash
cd plan_6_stock_volatility
uv sync --all-groups
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
uv run pytest -v
uv run ruff check .
```

### M1 Data Ingest & Day-1 Audit CLI

Rebuild interim macro/event tables and Day-1 & cross-period coverage audits deterministically offline from committed `data/raw/` snapshots (zero network calls):

```bash
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run python scripts/01_download_and_audit.py --skip-download
```

### M2 Panel Build, Purged Splits & SHA-256 Freeze CLI

Verify the cryptographic SHA-256 freeze manifest or deterministically refreeze offline from committed M1 inputs:

```bash
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run python scripts/02_build_panel_and_freeze.py --check
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run python scripts/02_build_panel_and_freeze.py --refreeze
```

### M3 Cross-Cycle EDA & H1 Panel HAR-RV CLIs

Run EDA tables and H1 Stock-FE Extended Panel HAR-RV with Driscoll–Kraay SE deterministically from the verified frozen bundle:

```bash
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run python scripts/03_eda.py
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run python scripts/04_panel_har_rv.py
```

### M4 H2 Forecasting, Calibration, HLN-DM & TreeSHAP CLI

Run all 4 models (Persistence, per-stock GJR-GARCH(1,1), Predictive HAR-RV, Global LightGBM), 2024 Val OOF calibration, single-shot 2025 Holdout Test evaluation, `h=5` HLN-DM + Holm tests, and Train-background TreeSHAP deterministically:

```bash
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run python scripts/05_forecast_backtest.py
```

### M5 Report Build & One-Command Phase A Replay CLIs

Build the 300-DPI Figure 1 & Figure 2, unified Table 1, CS482 Phase A Draft PDF, and automated 51-claim audit—or replay the entire Phase A pipeline deterministically offline from committed snapshots:

```bash
# Build figures, Table 1, CS482 Phase A Draft PDF, and run automated audit:
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run python scripts/06_write_report.py
# Or via the unified replay CLI (report-only fast rebuild vs full offline replay):
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run python scripts/run_phase_a.py --report-only
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run python scripts/run_phase_a.py --skip-download
```

### M6 5-Tier E2E & White-Box Adversarial Hardening Suite

Run the standalone 5-tier E2E & single-fault adversarial hardening suite (31 tests across Tiers 1–5; see [`TEST_READY.md`](TEST_READY.md)) or the full 97-test repository suite:

```bash
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run pytest tests/e2e --project-root . -v
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run pytest -v
```

## Data Sources & Anti-Leakage Protocol Summary

- **Deterministic 60-Stock Cross-Sector Universe**: 5 GICS sectors (`Technology`, `Financials`, `Healthcare`, `Consumer`, `Energy`) × 12 pre-2019 liquid S&P 500 constituents (Consumer split 6 Staples + 6 Discretionary under pre-2019 GICS taxonomy; zero 2025 lookahead).
- **Market & Macro Data**: Daily OHLCV and adjusted close for the 60 equities plus `SPY` and `^VIX` via Yahoo Finance, joined onto the canonical 2019–2025 NYSE trading calendar with FRED `T10Y2Y` and `VIXCLS` using backward-only forward-fill (never backfill).
- **Corporate Event Backbone (Plan 6 §2.1 Deterministic Fallback)**: Public SEC EDGAR Form `8-K`, `8-K/A`, `10-Q`, and `10-K` submissions across 2019–2025 with recursive `filings.files` archive pagination, academic `User-Agent`, `<= 8 req/s` throttling, DST-aware `16:00 America/New_York` cutoff (`13:00 ET` on all 15 NYSE early-close days; at/after close or non-trading-day filings mapped to the next NYSE trading day `t+1`), deduplication, and a hard cap of `<= 5` latest items per stock-day.
- **Purged + 5-Day Embargo Splits** (frozen verbatim for Phase A & Phase B):
  - `Train`: `2019-01-02` to `2023-12-20` (1,252 trading days; 1,231 modeling-complete days from `2019-02-01`)
  - `Embargo 1`: `2023-12-21` to `2023-12-29` (6 trading days)
  - `Val`: `2024-01-02` to `2024-12-20` (246 trading days; hyperparameter tuning & probability calibration only)
  - `Embargo 2`: `2024-12-23` to `2024-12-31` (6 trading days)
  - `Test`: `2025-01-02` to `2025-12-31` (250 trading days; 245 complete `t+1..t+5` target days; evaluated once)

## Directory Layout

```text
plan_6_stock_volatility/
├── .gitignore                          # [M0] Local cache, archive, weight, and LaTeX ignores
├── README.md                           # [M0–M6] Project overview, setup, and non-investment notice
├── TEST_READY.md                       # [M6] Final E2E Tiers 1–5 & adversarial sign-off guide
├── pyproject.toml                      # [M0] uv + hatchling package, ruff, and pytest configuration
├── uv.lock                             # [M0] Locked dependency graph
├── docs/
│   └── milestones/                     # [M1–M6] Tracked milestone delivery & AI execution docs
│       ├── README.md                   # [M1–M6] Master ledger, policy, and M2–M6 template
│       ├── M0_repo_cleanup_and_scaffold.md
│       ├── M1_data_ingest_and_day1_audit.md
│       ├── M2_panel_splits_and_freeze.md
│       ├── M3_eda_and_h1_panel_har_rv.md
│       ├── M4_h2_forecasting_calibration_and_shap.md
│       ├── M5_figures_table1_and_cs482_report.md
│       └── M6_e2e_and_adversarial_hardening.md
├── src/stock_volatility/
│   ├── __init__.py                     # [M0] Package metadata
│   ├── config.py                       # [M0] Canonical paths, constants, UNIVERSE_60, helpers
│   ├── ingest/                         # [M1,M6] Universe, OHLCV, macro, SEC EDGAR, Day-1 audit
│   ├── panel/                          # [M2] 16:00/13:00 ET alignment, features, splits, SHA-256 freeze
│   ├── eda.py                          # [M3] Volatility regimes, leverage, coverage, VIX x news
│   ├── regression.py                   # [M3] H1 Stock-FE Panel HAR-RV + Driscoll-Kraay SE
│   ├── baselines.py                    # [M4] Persistence, GJR-GARCH(1,1), HAR-RV baselines
│   ├── forecast.py                     # [M4] Global LightGBM + 2024 Val calibration
│   ├── evaluation.py                   # [M4] QLIKE/PR-AUC/ECE, block bootstrap, h=5 HLN-DM/Holm
│   ├── explain.py                      # [M4] Train-background TreeSHAP explainability
│   ├── figures.py                      # [M5] 300-DPI Figure 1 & Figure 2 (PNG + PDF)
│   └── report.py                       # [M5,M6] Table 1, deterministic PDF build & hardened audit
├── scripts/
│   ├── 01_download_and_audit.py        # [M1] Data ingest & offline --skip-download Day-1 audit CLI
│   ├── 02_build_panel_and_freeze.py    # [M2] Panel build, purged splits & --check/--refreeze CLI
│   ├── 03_eda.py                       # [M3] Cross-cycle EDA tables & summary CLI
│   ├── 04_panel_har_rv.py              # [M3] H1 Stock-FE Extended Panel HAR-RV CLI
│   ├── 05_forecast_backtest.py         # [M4] H2 baselines, LightGBM, HLN-DM & TreeSHAP CLI
│   ├── 06_write_report.py              # [M5] 300-DPI figures, Table 1, CS482 PDF & audit CLI
│   └── run_phase_a.py                  # [M5,M6] One-command offline Phase A replay CLI (--project-root)
├── data/                               # [M1–M2] raw/, interim/, frozen/ (FREEZE_MANIFEST.json)
├── outputs/                            # [M1–M5] figures/, metrics/, tables/
├── report/                             # [M5] phase_a_draft.tex, references.bib, PDF, audit
└── tests/
    ├── conftest.py                     # [M1,M6] MPLBACKEND, OMP_NUM_THREADS, block_network, --project-root
    ├── test_smoke.py                   # [M0] Scaffold, universe, split, library, TeX/Poppler checks
    ├── test_ingest.py                  # [M1] Unit, offline replay, artifact & milestone doc tests
    ├── test_panel_and_freeze.py        # [M2] Cutoff/early-close, anti-leakage, split & freeze tests
    ├── test_eda.py                     # [M3] Cross-cycle EDA & non-causal wording tests
    ├── test_regression.py              # [M3] Driscoll-Kraay equivalence, Holm & H1 contract tests
    ├── test_forecast.py                # [M4] Purged folds, GJR-GARCH, calibration & selection tests
    ├── test_evaluation_and_shap.py     # [M4] QLIKE/ECE, HLN-DM, bootstrap & TreeSHAP tests
    ├── test_figures_and_report.py      # [M5] 300-DPI figures, Table 1, A4 PDF audit & CLI tests
    └── e2e/                            # [M6] 5-tier opaque-box E2E (Tiers 1–4) & Tier 5 adversarial suite
```
