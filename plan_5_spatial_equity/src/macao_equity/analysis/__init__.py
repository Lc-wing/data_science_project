"""Milestone M2 spatial analytics: 2SFCA, inequality, LISA, regression, and MAUP."""

from macao_equity.analysis.inequality import (
    bh_fdr,
    bootstrap_gini_theil,
    bootstrap_indices,
    holm_bonferroni,
    lorenz_points,
    paired_gini_diff_bootstrap,
    weighted_gini,
    weighted_theil_t,
)
from macao_equity.analysis.maup import maup_r8_vs_r9
from macao_equity.analysis.regression import fit_prespecified_models, ols_hc3
from macao_equity.analysis.sfca import (
    composite_accessibility,
    gaussian_2sfca,
    gaussian_weight,
    prepare_walk_network,
    travel_minutes_from_graph,
)
from macao_equity.analysis.spatial_stats import (
    bridge_queen_weights,
    global_moran,
    island_safe_knn_weights,
    knn_weights,
    lisa_fdr,
    weights_diagnostics,
)

__all__ = [
    "bh_fdr",
    "bootstrap_gini_theil",
    "bootstrap_indices",
    "bridge_queen_weights",
    "composite_accessibility",
    "fit_prespecified_models",
    "gaussian_2sfca",
    "gaussian_weight",
    "global_moran",
    "holm_bonferroni",
    "island_safe_knn_weights",
    "knn_weights",
    "lisa_fdr",
    "lorenz_points",
    "maup_r8_vs_r9",
    "ols_hc3",
    "paired_gini_diff_bootstrap",
    "prepare_walk_network",
    "travel_minutes_from_graph",
    "weighted_gini",
    "weighted_theil_t",
    "weights_diagnostics",
]
