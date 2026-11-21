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

Route A (from-scratch GATv2, TransformerConv, and the ~5k GraphSAGE ablation, 10 seeds, inductive and transductive) runs on CPU:

```bash
export MPLBACKEND=Agg
uv run python scripts/run_phase_b_route_a.py
uv run pytest tests/test_phase_b.py
```

Route B (`BAAI/bge-m3` and `Qwen/Qwen3-Embedding-4B` QLoRA `r=8`) is the Colab L4/A100 path in `notebooks/phase_b_colab.ipynb`. The CPU entry refuses to download those weights. See `PHASE_B.md`.
