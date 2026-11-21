# data_science_project

CS482 data science project. Two plans are implemented here, each in its own folder with its own code, data, outputs, tests, notebooks, report, and `uv` environment.

| Folder | Plan | Status |
| --- | --- | --- |
| [`plan_5_spatial_equity/`](plan_5_spatial_equity/README.md) | [Plan 5](plans/plan_5_macao_facilities_ev_graph_dl.md): spatial equity of Macao public facilities and EV charging (`macao_equity`) | Phase A frozen, Phase B run, CS482 report built |
| [`plan_2_tourism_hotel/`](plan_2_tourism_hotel/README.md) | [Plan 2](plans/plan_2_macao_tourism_hotel_multivariate_dl.md): post-pandemic visitor mix and hotel occupancy by star tier | Phase A not started |

Shared material stays at the repository root:

- `plans/` holds the five plans and the review report (`plans/00_review_report.md`) whose constraints both implementations follow.
- `.gitignore` covers both folders.

## Working in one plan

Each plan folder is a separate `uv` project. Change into the folder first; every command and every relative path in that folder's documents starts there.

```bash
cd plan_5_spatial_equity      # or plan_2_tourism_hotel
uv sync --all-groups
export MPLBACKEND=Agg
uv run pytest
uv run ruff check .
```

Plan 5 also needs `pdflatex`, `bibtex`, and Poppler (`pdfinfo`, `pdftotext`) for the report checks. See `plan_5_spatial_equity/TEST_READY.md`.
