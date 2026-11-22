# Plan 5 — MacaoSpatialEquity

Implementation of [`plans/plan_5_macao_facilities_ev_graph_dl.md`](../plans/plan_5_macao_facilities_ev_graph_dl.md): spatial equity of Macao public facilities and EV charging (Phase A), plus the Phase B graph module.

This folder is a self-contained `uv` project. Run every command below from `plan_5_spatial_equity/`; every path in this folder's documents (`data/frozen/`, `outputs/`, `report/`, `scripts/`, `tests/`) is relative to it. The plans and the review report are shared and stay in `../plans/`.

## Phase A replay

MacaoSpatialEquity Phase A freezes the H3 grids, the 10-minute walk graph, and the five spatial blocks, then writes the accessibility maps, Table 1, and the CS482 draft from those inputs.

```bash
uv sync
export MPLBACKEND=Agg
uv run python scripts/run_phase_a.py --skip-download
```

`--skip-download` replays from the files already in `data/frozen/` and `data/interim/`. It does not download DSEC, CEM, or OpenStreetMap, and it does not rewrite the M1 text contracts (`graph_edges.csv`, `spatial_folds.csv`, `feature_manifest.json`). The M4 step rebuilds `data/frozen/cells.gpkg`, `data/frozen/cells_res8.gpkg`, `data/frozen/FREEZE_MANIFEST.json`, the figures in `outputs/figures/`, Table 1, and `report/CS482_GroupX_MacaoSpatialEquity_PhaseA_Draft.pdf`.

Quoted Gini, Theil, regression, and cross-validation numbers are copied from `outputs/tables/` and `outputs/metrics/phase_a_ml_baselines.json`. `report/go_nogo_audit.md` checks each reported figure against that source cell.

## Phase B

Route A (from-scratch edge-weighted `GATv2Conv`, `TransformerConv`, and unweighted ~5k `GraphSAGE` ablation; 10 seeds `42..51`, inductive and transductive on frozen 10-minute walk edges) runs on CPU:

```bash
export MPLBACKEND=Agg
uv run python scripts/run_phase_b_route_a.py
uv run pytest tests/test_phase_b.py
```

Route B (`BAAI/bge-m3` and `Qwen/Qwen3-Embedding-4B` frozen + QLoRA `r=8` fit strictly on `loss_mask`) is the Colab L4/A100 path in `notebooks/phase_b_colab.ipynb` (`scripts/colab_phase_b.py --allow-weight-download`, writing `*_run3.*` and `outputs/colab_runs/run3/` while preserving `outputs/colab_runs/run1/` and `outputs/colab_runs/run2/`). The CPU entry refuses to download large encoder weights. See `PHASE_B.md`.

Exploratory gradient-boosted tree baselines (XGBoost, LightGBM, HistGradientBoosting, outside the pre-specified plan) are stored in `outputs/metrics/phase_b_exploratory_trees.json` and `outputs/tables/phase_b_exploratory_trees.md`.

## Final CS482 report & verification

- Phase A draft (auto-generated and numerically audited): `report/CS482_GroupX_MacaoSpatialEquity_PhaseA_Draft.pdf` and `report/go_nogo_audit.md`.
- Final integrated CS482 report (Phase A + Phase B + exploratory trees): `report/CS482_GroupX_MacaoSpatialEquity.tex` / `report/CS482_GroupX_MacaoSpatialEquity.pdf` (fill in real group number, member names, and student IDs on the title page before Moodle submission).
- Full lint and test suite (unit tests + E2E Tiers 1–5):

```bash
export MPLBACKEND=Agg
uv run ruff check .
uv run pytest -v
```
