# data_science_project

CS482 data science project. Three plans are implemented here, each in its own folder with its own code, data, outputs, tests, notebooks, report, and `uv` environment.

| Folder | Plan | Status |
| --- | --- | --- |
| [`plan_5_spatial_equity/`](plan_5_spatial_equity/README.md) | [Plan 5](plans/plan_5_macao_facilities_ev_graph_dl.md): spatial equity of Macao public facilities and EV charging (`macao_equity`) | Phase A frozen, Phase B run, CS482 report built |
| [`plan_2_tourism_hotel/`](plan_2_tourism_hotel/README.md) | [Plan 2](plans/plan_2_macao_tourism_hotel_multivariate_dl.md): post-pandemic visitor mix and hotel occupancy by star tier (`macao_tourism`) | Phase A frozen, backtest complete, CS482 Phase A draft built |
| [`plan_6_stock_volatility/`](plan_6_stock_volatility/README.md) | [Plan 6](plans/plan_6_stock_volatility_news_llm.md): US equity volatility and downside tail-risk forecasting with corporate events and LLM (`stock_volatility`) | Phase A in progress (M0–M2 complete, data & splits frozen; see [milestone delivery docs](plan_6_stock_volatility/docs/milestones/README.md)) |

Shared material stays at the repository root:

- `plans/` holds the nine plans (`plan_1`–`plan_9`) and the review report (`plans/00_review_report.md`) whose constraints all implementations follow.
- `.gitignore` covers all plan folders (each plan folder also keeps its own `.gitignore` for plan-specific caches and local build artifacts).

## Working in one plan

Each plan folder is a separate `uv` project. Change into the folder first; every command and every relative path in that folder's documents starts there.

```bash
cd plan_5_spatial_equity      # or plan_2_tourism_hotel, plan_6_stock_volatility
uv sync --all-groups
export MPLBACKEND=Agg
uv run pytest
uv run ruff check .
```

Plans 2, 5, and 6 also need `pdflatex`, `bibtex` (Plans 5 and 6), and Poppler (`pdfinfo`, `pdftotext`) to build and audit their CS482 PDF reports. See `plan_5_spatial_equity/TEST_READY.md` for Plan 5's report-check toolchain.
