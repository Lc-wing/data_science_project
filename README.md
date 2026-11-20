# data_science_project
a repository for finsishing my data science project

## Phase A replay

MacaoSpatialEquity Phase A freezes the H3 grids, the 10-minute walk graph, and the five spatial blocks, then writes the accessibility maps, Table 1, and the CS482 draft from those inputs.

```bash
uv sync
export MPLBACKEND=Agg
uv run python scripts/run_phase_a.py --skip-download
```

`--skip-download` replays from the files already in `data/frozen/` and `data/interim/`. It does not download DSEC, CEM, or OpenStreetMap, and it does not rewrite the M1 text contracts (`graph_edges.csv`, `spatial_folds.csv`, `feature_manifest.json`). The M4 step rebuilds `data/frozen/cells.gpkg`, `data/frozen/cells_res8.gpkg`, `data/frozen/FREEZE_MANIFEST.json`, the figures in `outputs/figures/`, Table 1, and `report/CS482_GroupX_MacaoSpatialEquity_PhaseA_Draft.pdf`.

Quoted Gini, Theil, regression, and cross-validation numbers are copied from `outputs/tables/` and `outputs/metrics/phase_a_ml_baselines.json`. `report/go_nogo_audit.md` checks each reported figure against that source cell.
