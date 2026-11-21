"""Pre-specified Phase B constants.

These values are fixed before any test-block score is computed. They are not a
search space. GPSConv is named in the plan beside TransformerConv; this module
uses TransformerConv and does not swap to GPSConv after seeing scores.
GNNExplainer is named beside attention weights; exported explanations use
GATv2 attention.
"""

from __future__ import annotations

from dataclasses import dataclass

# Project seed 42, then the next nine integers. The plan requires ten seeds and
# does not publish a different list.
PHASE_B_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)

PRIMARY_METRICS: tuple[str, ...] = ("roc_auc", "pr_auc", "top20_hit_rate")

PROTOCOLS: tuple[str, ...] = ("inductive", "transductive")
ROUTE_A_MODELS: tuple[str, ...] = ("gatv2", "graph_transformer", "graphsage_5k")

# Route A capacity, counted before any test score.
# GATv2: PyG out_channels=128 per head, 4 heads, concat. Two layers are about
# 1.06M parameters, inside the plan's 0.2M–1.5M cap (the table's "~0.5M" is the
# same family, not a second search).
# TransformerConv with concat at hidden=128 and 4 heads is about 2.1M, over the
# cap. Concat is therefore off so the four heads are averaged, hidden stays 128,
# and the stack stays near 0.43M. GPSConv is not used.
# GraphSAGE hidden=44 is the ~5k capacity ablation, outside the 128–256 band on
# purpose.
GATV2_HIDDEN = 128
GATV2_HEADS = 4
GATV2_CONCAT = True
TRANSFORMER_HIDDEN = 128
TRANSFORMER_HEADS = 4
TRANSFORMER_CONCAT = False
SAGE_HIDDEN = 44

PARAM_BUDGET_MIN = 200_000
PARAM_BUDGET_MAX = 1_500_000
SAGE_PARAM_MIN = 4_000
SAGE_PARAM_MAX = 6_000
TOTAL_PARAM_LIMIT = 5_000_000_000

DROPOUT = 0.5
DROPEDGE_P = 0.3
WEIGHT_DECAY = 1e-2
LEARNING_RATE = 1e-3
LEARNING_RATE_MIN = 1e-5
MAX_EPOCHS = 60
MIN_EPOCHS = 15
PATIENCE = 10

# Route B text arm. One LoRA seed keeps the Colab path inside the plan's
# 20–45 minute budget. The ten-seed distribution is applied to the GATv2 head,
# which is the small-graph variance the plan asks to report.
ROUTE_B_LORA_SEEDS: tuple[int, ...] = (42,)
ROUTE_B_GNN_SEEDS: tuple[int, ...] = PHASE_B_SEEDS
TEXT_DIM = 64
QWEN_MODEL_ID = "Qwen/Qwen3-Embedding-4B"
BGE_MODEL_ID = "BAAI/bge-m3"
QWEN_LORA_R = 8
QWEN_LORA_ALPHA = 16
QWEN_LORA_DROPOUT = 0.05
QWEN_LORA_TARGET_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
QWEN_LORA_MAX_STEPS = 40
QWEN_LORA_LR = 1e-4
QWEN_LORA_BATCH_SIZE = 16
TEXT_MAX_LENGTH = 256
TEXT_ENCODE_BATCH_SIZE = 16
QWEN_TASK_INSTRUCTION = (
    "Represent this grid cell's district and places for a spatial classification task."
)

# Candidate screen uses accessibility only after the model has scored cells.
# acc15_ev_charger is never a feature. The cut is the median over all cells,
# and the score cut is the 80th percentile among cells that currently have no
# charger. Both cuts are fixed here.
CANDIDATE_ACCESS_COLUMN = "acc15_ev_charger"
CANDIDATE_ACCESS_QUANTILE = 0.50
CANDIDATE_SCORE_QUANTILE = 0.80
CANDIDATE_MODEL = "gatv2"
CANDIDATE_PROTOCOL = "inductive"
ATTENTION_SEED = PHASE_B_SEEDS[0]
ATTENTION_TOP_K = 3

CANDIDATE_INTERPRETATION = (
    "Historical-configuration association. A high score in a cell that has no "
    "charger and low 15-minute EV accessibility is a field-check candidate, "
    "not a recommendation to install a charger."
)

PHASE_A_MODELS_IN_TABLE: tuple[str, ...] = (
    "naive_prior",
    "logistic_regression",
    "random_forest",
)

ROUTE_B_ARMS: tuple[str, ...] = (
    "bge_m3_gatv2",
    "qwen3_embedding_4b_frozen_gatv2",
    "qwen3_embedding_4b_lora_gatv2",
)


@dataclass(frozen=True)
class EpochBudget:
    """Fixed training length. Tests pass a shorter budget; the study uses SPEC."""

    max_epochs: int = MAX_EPOCHS
    min_epochs: int = MIN_EPOCHS
    patience: int = PATIENCE


SPEC_BUDGET = EpochBudget()
