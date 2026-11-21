# Plan 2 — Macao visitor mix and hotel occupancy by star tier

Implementation folder for [`plans/plan_2_macao_tourism_hotel_multivariate_dl.md`](../plans/plan_2_macao_tourism_hotel_multivariate_dl.md), under the constraints in [`plans/00_review_report.md`](../plans/00_review_report.md).

Everything for Plan 2 lives in this folder: code, data, outputs, tests, and (later) the report. Nothing here imports from, or writes to, `plan_5_spatial_equity/`.

## Phase A steps

| Step | Content | Status |
| --- | --- | --- |
| 1 | Data download and Day-1 audit | done |
| 2 | Panel build and freeze (splits, manifest, pre-registered analysis spec) | done |
| 3 | EDA (STL, change points, scatter by period) | done |
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

## Step 2: panel build and freeze

```bash
uv run python scripts/02_build_panel_and_freeze.py             # build; keeps an identical freeze
uv run python scripts/02_build_panel_and_freeze.py --check     # verify hashes and a rebuild, write nothing
uv run python scripts/02_build_panel_and_freeze.py --refreeze  # replace a freeze that changed
```

The script builds the analysis panel from `data/raw` and `data/interim`, writes it to `data/frozen/`, and records the SHA-256 of every frozen file in `data/frozen/FREEZE_MANIFEST.json`. Steps 3 to 6, and Phase B, read these files and nothing else. If a rebuild differs from the existing freeze the script stops and writes nothing unless `--refreeze` is given.

| File | Content |
| --- | --- |
| `data/frozen/panel_tier_month.csv` | 804 rows: 4 star classes × 201 months (2010-01 to 2026-09), 29 columns, no missing value |
| `data/frozen/months.csv` | The 201 months with the visitor, calendar, and typhoon variables |
| `data/frozen/holiday_periods.csv` | The 115 holiday periods with the day each schedule was published, and the 2 schedules a later notice replaced |
| `data/frozen/rolling_origins.csv` | 36 validation origins and 26 test origins, one month ahead |
| `data/frozen/split_metadata.json` | The split rules |
| `data/frozen/analysis_spec.json` | The analysis specification for steps 3 to 5 |
| `data/frozen/data_dictionary.md` | Every column with its unit and definition |

`outputs/tables/panel_variable_summary.csv` gives the range of every variable over the whole window. It is not split by period or star class: no comparison between periods or classes was looked at before the specification was fixed.

### Splits

- **Test:** origins 2024-07 to 2026-08, targets 2024-08 to 2026-09: 26 origins × 4 classes = 104 forecasts. Evaluated once, in step 5.
- **Validation:** targets 2018-01 to 2019-12 (24) and 2023-07 to 2024-06 (12). All lie before the first test origin. Orders and hyperparameters are chosen here.
- **Training rows:** target months up to the origin, from 2011-01 (twelve lags are needed), in the pre-pandemic and post-pandemic periods. A sensitivity run keeps the pandemic months.

### Analysis specification

[`data/frozen/analysis_spec.json`](data/frozen/analysis_spec.json) fixes, before any model is fitted: the regression equation, the standard errors, the H1 and H2 null hypotheses, the Holm families, the SARIMAX and LightGBM candidate grids, the selection rule, the metrics, the Diebold–Mariano comparisons, the H3 decision rule, and the sensitivity runs. Diagnostics are reported and change nothing. A departure from the file is listed with its reason where it happens; the file is not edited after results are seen.

### Holiday days known ahead

The panel carries the holiday counts twice. `cn_holiday_days` follows the final schedule and enters the ex-post regression. `cn_holiday_days_at_origin` follows the notices published by the end of the month before, and is the calendar feature of the forecasts. The two differ in four months: 2016-12, 2018-12, and 2022-12 (the notice for the next year, which sets the New Year period, came out in December) and 2020-01 (the Spring Festival period was extended on 27 January 2020).

## Step 3: exploratory data analysis

```bash
uv run python scripts/03_eda.py
```

The script reads `data/frozen/` only, and stops if any frozen file differs from the manifest. Everything it writes is exploratory description. Nothing it writes is an input to steps 4 and 5, and the period boundaries stay those of the plan.

| File | Content |
| --- | --- |
| [`outputs/tables/eda_summary.md`](outputs/tables/eda_summary.md) | All tables of the step, with the settings |
| `outputs/figures/fig1_visitor_mix_and_occupancy.{png,pdf}` | Figure 1: the three visitor-mix shares with their change points, and occupancy by star class |
| `outputs/figures/eda_stl_occupancy.{png,pdf}` | Occupancy with its trend, the seasonal profile, and the remainder, by star class |
| `outputs/figures/eda_scatter_occupancy_overnight.{png,pdf}` | Occupancy against log overnight visitors, by star class, coloured by period |
| `outputs/tables/eda_stl_*.csv` | STL components, strength of trend and seasonality, seasonal profiles |
| `outputs/tables/eda_change_point_*.csv` | Change points at each penalty, and the segments of the main run |
| `outputs/tables/eda_visitor_mix_by_period.csv`, `eda_occupancy_by_period.csv` | Means by period |
| `outputs/metrics/eda_summary.json` | The same numbers, with the SHA-256 of the panel and of the specification |

### Settings

- **STL on occupancy:** period 12, robust, as the specification says. One fit per star class and per block (pre-pandemic, post-pandemic), with a seasonal pattern that is constant inside a block. Pandemic months are not decomposed.
- **PELT on the three shares:** each share minus its STL seasonal component; piecewise-constant mean; segments of at least 12 months; penalty 2 × noise_sd² × ln(n), where noise_sd comes from the 12-month differences. The change points are also reported at half, twice, and four times that penalty, and without the seasonal adjustment.

One setting was changed after the first run. STL was first fitted once per star class over all 201 months. That fit put the pandemic collapse into the seasonal component of the months after it: for 5-star hotels in 2023-08 (occupancy 90.6%) it gave a seasonal component of −34.8 points. `outputs/tables/eda_stl_full_series_check.csv` keeps that result. The PELT settings were not changed.

## Layout

```text
plan_2_tourism_hotel/
├── pyproject.toml, uv.lock
├── src/macao_tourism/      # config, ingest, audit, panel, splits, spec, freeze, eda, plots
├── scripts/                # one script per Phase A step
├── tests/
├── data/raw/               # official snapshots + data_provenance.json
├── data/interim/           # parsed tables
├── data/frozen/            # analysis panel, splits, spec, FREEZE_MANIFEST.json
└── outputs/                # tables, metrics, figures
```
