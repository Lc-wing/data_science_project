# Plan 2 — Macao visitor mix and hotel occupancy by star tier

Implementation folder for [`plans/plan_2_macao_tourism_hotel_multivariate_dl.md`](../plans/plan_2_macao_tourism_hotel_multivariate_dl.md), under the constraints in [`plans/00_review_report.md`](../plans/00_review_report.md).

Everything for Plan 2 lives in this folder: code, data, outputs, tests, and (later) the report. Nothing here imports from, or writes to, `plan_5_spatial_equity/`.

## Phase A steps

| Step | Content | Status |
| --- | --- | --- |
| 1 | Data download and Day-1 audit | done |
| 2 | Panel build and freeze (splits, manifest, pre-registered analysis spec) | not started |
| 3 | EDA (STL, change points, scatter by period) | not started |
| 4 | H1/H2 panel fixed-effects regression with Driscoll–Kraay standard errors | not started |
| 5 | H3 rolling-origin backtest (seasonal naive, SARIMAX, LightGBM) | not started |
| 6 | Figures, tables, Phase A report draft, one-command replay | not started |

Phase B is out of scope here.

## Setup

```bash
cd plan_2_tourism_hotel
uv sync --all-groups
export MPLBACKEND=Agg
```

## Step 1: data download and Day-1 audit

```bash
uv run python scripts/01_download_and_audit.py                  # download, then audit
uv run python scripts/01_download_and_audit.py --skip-download  # audit the saved snapshots
uv run pytest
uv run ruff check .
```

The audit is in [`outputs/tables/day1_data_audit.md`](outputs/tables/day1_data_audit.md); the same numbers are in `outputs/metrics/day1_data_audit.json`.

### Sources

All data are official publications, downloaded as published. No value is simulated or imputed.

| Data | Source | Raw snapshot |
| --- | --- | --- |
| Monthly visitor arrivals (total, same-day, overnight, by place of travel document, by mode of entry, average length of stay) | DSEC Time Series Database, the database behind `dsec.gov.mo/en-US/Statistic?id=402` | `data/raw/dsec_tsdb/` |
| Monthly occupancy rate, guest rooms, operating establishments, and guests by star class; package-tour visitors | DSEC Time Series Database, `id=403` | `data/raw/dsec_tsdb/` |
| Latest monthly statistical tables (`.xlsx`), used to check the database values | DSEC releases for `id=402` and `id=403` | `data/raw/dsec_releases/` |
| Mainland public-holiday schedule, 2010–2026 | State Council notices on gov.cn, one per year, plus three amending notices | `data/raw/cn_holiday_notices/` |
| Tropical cyclone warning signals, 2010–2026 | SMG tropical cyclone reports, annual signal table, monthly climate summaries | `data/raw/smg_tropical_cyclone/` |

`data/raw/data_provenance.json` records the URL, download time, size, and SHA-256 of every raw file.

The SMG reports and monthly summaries are PDFs (about 160 MB in total). They are downloaded to `.cache/` and are not committed. The repository keeps the text extracted from each PDF, and the provenance file keeps each PDF's URL, size, and SHA-256. `--skip-download` works from the committed text.

DICJ gaming revenue is optional in the plan and is not used.

### Definitions fixed in this step

- **Star classes:** DSEC's `5-star hotels`, `4-star hotels`, `3-star hotels`, `2-star hotels`, as published. No class is merged or re-labelled.
- **Holiday period:** the days a State Council notice declares off, joined with adjoining Saturdays and Sundays that the notice does not declare as working days. Where a notice states the length of a holiday, the parsed period must have that length, or the run stops.
- **Typhoon day:** a calendar day (Macao time) on which tropical cyclone signal No. 8 or higher was in force for any length of time.

### Outputs

| File | Content |
| --- | --- |
| `data/interim/dsec_monthly_long.csv` | All DSEC series, one row per series and month |
| `data/interim/dsec_indicator_registry.csv` | Indicator ids with DSEC's own label, unit, and path |
| `data/interim/cn_holiday_periods.csv`, `cn_holiday_days_monthly.csv` | Holiday periods and holiday days per month |
| `data/interim/smg_signal_events.csv`, `smg_signal_days.csv`, `smg_signal_days_monthly.csv` | Signal records, days with a signal, and typhoon days per month |
| `outputs/tables/day1_*.csv` | Coverage, release cross-check, star-class continuity, SMG cross-checks |

## Layout

```text
plan_2_tourism_hotel/
├── pyproject.toml, uv.lock
├── src/macao_tourism/      # config, ingest (dsec, holidays, smg), audit
├── scripts/                # one script per Phase A step
├── tests/
├── data/raw/               # official snapshots + data_provenance.json
├── data/interim/           # parsed tables
└── outputs/                # tables, metrics
```
