"""Post-hoc field-check list and a qualitative attention export.

Accessibility enters only here, after scores exist. It is not a model feature.
The list is an association screen under the historical charger configuration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from macao_equity.phase_b.spec import (
    ATTENTION_TOP_K,
    CANDIDATE_ACCESS_QUANTILE,
    CANDIDATE_INTERPRETATION,
    CANDIDATE_SCORE_QUANTILE,
)
from macao_equity.phase_b.train import FoldFit


def select_candidates(
    cell_id: np.ndarray,
    y: np.ndarray,
    acc15_ev: np.ndarray,
    mean_probability: np.ndarray,
) -> pd.DataFrame:
    """Cells with no charger, low accessibility, and a high mean score."""
    labels = np.asarray(y, dtype=int)
    access = np.asarray(acc15_ev, dtype=float)
    scores = np.asarray(mean_probability, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("candidate scores must be finite")
    no_charger = labels == 0
    if not np.any(no_charger):
        raise ValueError("candidate screen needs cells without a charger")
    access_cut = float(np.quantile(access, CANDIDATE_ACCESS_QUANTILE))
    score_cut = float(np.quantile(scores[no_charger], CANDIDATE_SCORE_QUANTILE))
    keep = no_charger & (access <= access_cut) & (scores >= score_cut)
    frame = pd.DataFrame(
        {
            "cell_id": np.asarray(cell_id)[keep],
            "mean_predicted_probability": scores[keep],
            "acc15_ev_charger": access[keep],
            "has_ev_charger": labels[keep],
            "access_cut": access_cut,
            "score_cut": score_cut,
            "interpretation": CANDIDATE_INTERPRETATION,
        }
    )
    frame = frame.sort_values(
        ["mean_predicted_probability", "cell_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return frame


def attention_neighbor_rows(
    cell_id: np.ndarray,
    fit: FoldFit,
    *,
    top_k: int = ATTENTION_TOP_K,
) -> list[dict[str, Any]]:
    """Top incoming GATv2 neighbors for each test cell, self-loops removed."""
    if fit.attention_edges is None or fit.attention_weights is None:
        return []
    src = fit.attention_edges[0]
    dst = fit.attention_edges[1]
    weights = fit.attention_weights
    ids = np.asarray(cell_id).astype(str)
    rows: list[dict[str, Any]] = []
    test_nodes = np.where(np.isfinite(fit.probabilities))[0]
    for node in test_nodes:
        incoming = np.flatnonzero((dst == int(node)) & (src != int(node)))
        if incoming.size == 0:
            continue
        order = sorted(
            incoming.tolist(),
            key=lambda index: (-float(weights[index]), int(src[index])),
        )
        for rank, index in enumerate(order[:top_k], start=1):
            rows.append(
                {
                    "model": fit.model_name,
                    "protocol": fit.protocol,
                    "seed": fit.seed,
                    "fold": fit.fold,
                    "cell_id": ids[int(node)],
                    "neighbor_cell_id": ids[int(src[index])],
                    "mean_attention": float(np.round(float(weights[index]), 8)),
                    "rank": rank,
                    "note": "Qualitative model behavior only. Not a causal neighbor effect.",
                }
            )
    return rows
