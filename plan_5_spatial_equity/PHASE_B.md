# Phase B — graph module (Colab path)

Plan 5 lives in the `plan_5_spatial_equity/` folder of the repository. Run every command in this file from that folder; every path here is relative to it.

Phase A is already frozen. This module only **reads** `data/frozen/`. It does not rewrite those files, and it does not refit the Phase A logistic or random-forest numbers.

The task is node classification of `has_ev_charger` on the 287 populated H3 resolution-9 cells. Model inputs are exactly `feature_manifest.json` → `ml_features`. Excluded from the feature matrix: EV counts and capacity, `acc10_*`, `acc15_*`, `lisa_*`, POI text, and coordinates. CRS remains EPSG:32649. Scores are associations under the frozen spatial blocks. They are not a causal effect and they are not a recommendation to install a charger.

## What already runs on CPU

```bash
uv sync --group dev
export MPLBACKEND=Agg
uv run python scripts/run_phase_b_route_a.py
uv run pytest tests/test_phase_b.py
```

Route A trains three pre-specified networks for 10 seeds (`42` through `51`), both protocols, and the five frozen blocks:

| Model | Spec | Role |
| --- | --- | --- |
| `gatv2` | 2× `GATv2Conv` (`edge_dim=1`, `fill_value=1.0`), hidden 128, 4 heads, concat, LayerNorm, ELU, residual | Main Route A model (~1.06M) |
| `graph_transformer` | 2× `TransformerConv` (`edge_dim=1`), hidden 128, 4 heads, heads averaged so the stack stays under 1.5M, LayerNorm, ELU, residual | The plan's TransformerConv, not GPSConv |
| `graphsage_5k` | 2× `SAGEConv` (unweighted), hidden 44 | ~5k capacity ablation |

Shared training choices, fixed before any test score: Dropout 0.5, DropEdge 0.3 on undirected pairs, AdamW weight decay `1e-2`, cosine learning rate `1e-3` → `1e-5`, at most 60 epochs, early stop on the deterministic peninsula inner block (block 2, or block 1 when fold == 2; patience 10, not before epoch 15) so every fold trains on both Peninsula and Island blocks. Walk-time inverse weights (`weight = 1 / walk_time_min`) are normalized strictly by the training-edge mean. The outer test block is never in the loss and never chooses the epoch.

- **Inductive:** training removes test nodes and every edge that touches them. At evaluation, edges keep `train -> train`, `train -> test`, and `test -> test` while dropping `test -> train`, so test cells aggregate spatial neighborhood features without leaking back into training representations.
- **Transductive:** training keeps cross-block edges. Test labels stay masked. Test features can enter messages. That is the sensitivity the plan asks for.

Primary metrics are ROC-AUC, PR-AUC, and the top-20% hit rate (same definition as Phase A: recall inside the top `ceil(0.20 * n)` scores). The Route A table cell is the mean across seeds of the five-fold mean, with a sample standard deviation across seeds. The Phase A rows copy the published fold mean and fold standard deviation for seed 42. Those two ± quantities are not the same dispersion.

Outputs:

- `outputs/metrics/phase_b_route_a.json`
- `outputs/metrics/phase_b_graph_assembly.json`
- `outputs/tables/phase_b_route_a_folds.csv`
- `outputs/tables/phase_b_route_a_oof.csv`
- `outputs/tables/phase_b_comparison.csv`
- `outputs/tables/phase_b_comparison.md`
- `outputs/tables/phase_b_candidates.csv`
- `outputs/tables/phase_b_attention_neighbors.csv`

`acc15_ev_charger` is used only in the candidate screen, after scores exist. A candidate has no charger, accessibility at or below the all-cell median, and a mean inductive GATv2 probability at or above the 80th percentile among cells with no charger. The attention file is a qualitative neighbor list for seed 42, not an importance claim.

## What runs on Colab (L4 or A100)

This VM does not download `Qwen/Qwen3-Embedding-4B` or `BAAI/bge-m3`. `scripts/colab_phase_b.py` exits before any hub call unless it is started with `--allow-weight-download` **and** CUDA is available.

Open `notebooks/phase_b_colab.ipynb` in Colab (GitHub path `plan_5_spatial_equity/notebooks/phase_b_colab.ipynb`). Its first cell clones `main`, or pulls `main` if the repository is already cloned, and changes into `plan_5_spatial_equity/`. The same steps by hand:

```bash
git clone --branch main https://github.com/Lc-wing/data_science_project.git
cd data_science_project/plan_5_spatial_equity
pip install -q h3 torch-geometric transformers peft bitsandbytes accelerate
python scripts/colab_phase_b.py --allow-weight-download
```

`scripts/colab_phase_b.py` resolves `data/frozen/` and `outputs/` from its own location, so it reads and writes inside `plan_5_spatial_equity/` whatever the working directory is. The notebook's last cell reads `outputs/tables/phase_b_comparison_run3.md` by relative path, which is why the first cell changes directory.

Do not rerun `scripts/run_phase_b_route_a.py` on Colab; updated Route A metrics are already committed in the repo. The Colab command writes **run 3** only:

- `outputs/metrics/phase_b_route_b_run3.json`
- `outputs/tables/phase_b_route_b_folds_run3.csv`
- `outputs/tables/phase_b_comparison_run3.csv`
- `outputs/tables/phase_b_comparison_run3.md`
- a copy under `outputs/colab_runs/run3/`

It does not replace `outputs/colab_runs/run1/`, `outputs/colab_runs/run2/`, or earlier comparison tables.

`h3` is imported as soon as `scripts/colab_phase_b.py` loads `macao_equity.ml.baselines`. A stock Colab runtime already provides geopandas, shapely, pyproj, numpy, pandas, and scikit-learn. `torch-geometric` is the other import-time package that runtime does not include. `transformers`, `peft`, `bitsandbytes`, and `accelerate` are used when the weight download actually starts.

Route B text arms (Run 3), each with an edge-weighted GATv2 head and TransformerConv head, both protocols, seeds 42–51:

1. Numeric GATv2 and TransformerConv — copied from Route A, not trained again.
2. Frozen `BAAI/bge-m3`. The tokenizer pads on the right, and the CLS vector is the first non-pad token. PCA to 64 dimensions is fit on the GNN `loss_mask` rows only.
3. Frozen `Qwen/Qwen3-Embedding-4B` + MRL truncation to 64 dimensions, then L2 normalization.
4. QLoRA (`r=8`, alpha 16, dropout 0.05, targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, 4-bit NF4, 4 shuffled epochs without replacement, cosine LR `1e-4` for LoRA / `1e-3` for classification head, grad clip `1.0`, seed 42) fit **strictly on `loss_mask` rows** (both the outer test block AND the inner validation block stay out, eliminating Run 2's validation-leakage bug). The JSON records the checkpoint revision, unpacked logical parameter count (`effective_parameter_count`, ~4.03B ≤ 5B), trainable parameter count, and first/last epoch losses.

The ten seeds apply to the graph heads. The 4B LoRA encoder uses one pre-specified seed. Total logical parameters of each encoder must stay ≤ 5B.
