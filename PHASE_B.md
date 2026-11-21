# Phase B — graph module (Colab path)

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
| `gatv2` | 2× `GATv2Conv`, hidden 128, 4 heads, concat, LayerNorm, residual | Main Route A model (~1.06M) |
| `graph_transformer` | 2× `TransformerConv`, hidden 128, 4 heads, heads averaged so the stack stays under 1.5M | The plan's TransformerConv, not GPSConv |
| `graphsage_5k` | 2× `SAGEConv`, hidden 44 | ~5k capacity ablation |

Shared training choices, fixed before any test score: Dropout 0.5, DropEdge 0.3 on undirected pairs, AdamW weight decay `1e-2`, cosine learning rate `1e-3` → `1e-5`, at most 60 epochs, early stop on the inner block with the largest id (patience 10, not before epoch 15). The outer test block is never in the loss and never chooses the epoch.

- **Inductive:** training removes test nodes and every edge that touches them. At evaluation, edges run from training nodes to test nodes only.
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

## What runs on Colab (L4 or A100, about 20–45 minutes)

This VM does not download `Qwen/Qwen3-Embedding-4B` or `BAAI/bge-m3`. `scripts/colab_phase_b.py` exits before any hub call unless it is started with `--allow-weight-download` **and** CUDA is available.

On Colab, open `notebooks/phase_b_colab.ipynb` or run:

```bash
pip install -q h3 torch-geometric transformers peft bitsandbytes accelerate
python scripts/run_phase_b_route_a.py   # skip if the CPU metrics are already in the repo
python scripts/colab_phase_b.py --allow-weight-download
```

`h3` is imported as soon as `scripts/colab_phase_b.py` loads `macao_equity.ml.baselines`. A stock Colab runtime already provides geopandas, shapely, pyproj, numpy, pandas, and scikit-learn. `torch-geometric` is the other import-time package that runtime does not include. `transformers`, `peft`, `bitsandbytes`, and `accelerate` are used when the weight download actually starts.

Route B arms, same blocks and the same GATv2 head:

1. Numeric GATv2 — copied from Route A, not trained again.
2. Frozen `BAAI/bge-m3` (CLS pool) + PCA to 64 dimensions fit on the loss-mask rows only.
3. Frozen `Qwen/Qwen3-Embedding-4B` + MRL truncation to 64 dimensions, then L2 normalization.
4. QLoRA (`r=8`, alpha 16, dropout 0.05, targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, 4-bit NF4, 40 steps, seed 42) fit on the loss mask only, then the same 64-dimensional export.

The ten seeds apply to the GATv2 head. The 4B LoRA encoder uses one pre-specified seed so the Colab run stays inside the plan's 20–45 minute budget. Repeating LoRA across ten seeds is not a second model choice; it is left out of this budget on purpose. The script records the checkpoint revision, the LoRA config, and both seed lists. Total parameters of the encoder must stay ≤ 5B.

Route B writes `outputs/metrics/phase_b_route_b.json` and refreshes the comparison table. Until that file exists, the Route B rows are `not_run` with an explicit blocker. They are not removed.
