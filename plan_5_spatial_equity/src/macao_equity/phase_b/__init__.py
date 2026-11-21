"""Phase B graph module for Plan 5.

Route A trains the pre-specified from-scratch GNNs on CPU. Route B is the Colab
path for BGE-M3 and Qwen3-Embedding-4B. Neither route treats accessibility,
coordinates, or EV counts as charger-prediction features.
"""

from macao_equity.phase_b.spec import PHASE_B_SEEDS, PRIMARY_METRICS

__all__ = ["PHASE_B_SEEDS", "PRIMARY_METRICS"]
