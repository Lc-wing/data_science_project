"""Phase B scores. The primary triple is the plan's comparison table."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score, roc_auc_score

from macao_equity.ml.baselines import top20_hit_rate
from macao_equity.phase_b.spec import PRIMARY_METRICS


def _round8(value: float) -> float:
    return float(np.round(float(value), 8))


def score_binary(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """ROC-AUC, PR-AUC, and top-20% hit rate. No other primary metric."""
    y = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        roc = float(roc_auc_score(y, scores))
        pr_auc = float(average_precision_score(y, scores))
    scored = {
        "roc_auc": _round8(roc),
        "pr_auc": _round8(pr_auc),
        "top20_hit_rate": _round8(top20_hit_rate(y, scores)),
    }
    missing = [name for name in PRIMARY_METRICS if name not in scored]
    if missing:
        raise RuntimeError(f"primary metrics missing: {missing}")
    return scored


def mean_std(values: list[float] | np.ndarray, *, ddof: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("cannot summarize an empty metric list")
    mean = float(arr.mean())
    if arr.size == 1:
        std = 0.0
    else:
        std = float(arr.std(ddof=ddof))
    return _round8(mean), _round8(std)
