"""Spatial-block classification baselines for ``has_ev_charger`` on H3 r9.

Inputs are exactly the manifest ``ml_features``. Each outer model is an sklearn
``Pipeline`` fitted only on the other frozen blocks, with hyperparameter search
restricted to those same blocks. Reported scores are predictive associations
under that split. They are not causal effects and are not a siting policy.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES: tuple[str, ...] = ("naive_prior", "logistic_regression", "random_forest")
METRIC_NAMES: tuple[str, ...] = ("roc_auc", "pr_auc", "brier", "top20_hit_rate")

# Inner spatial search only. The outer test block is never in this grid search.
LOGISTIC_C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
RF_MAX_DEPTH_GRID: tuple[int | None, ...] = (4, None)
RF_MIN_SAMPLES_LEAF_GRID: tuple[int, ...] = (2, 8)
RF_N_ESTIMATORS = 200

_BANNED_EXACT = frozenset(
    {
        "centroid_x",
        "centroid_y",
        "geometry",
        "poi_text",
        "lat",
        "lon",
        "latitude",
        "longitude",
    }
)
_EV_LEAK_RE = re.compile(
    r"(?i)(^|_)(ev|evse|charger|charging|tesla|pile|pilenum|total_kw|"
    r"acc10|acc15|lisa|target|spatial_block|fold)(_|$)|2sfca|charg"
)

CV_COLUMNS: tuple[str, ...] = (
    "model",
    "mean_roc_auc",
    "std_roc_auc",
    "mean_pr_auc",
    "std_pr_auc",
    "mean_brier",
    "std_brier",
    "mean_top20_hit_rate",
    "std_top20_hit_rate",
    "oof_roc_auc",
    "oof_pr_auc",
    "oof_brier",
    "oof_top20_hit_rate",
)
OOF_COLUMNS: tuple[str, ...] = (
    "cell_id",
    "spatial_block",
    "has_ev_charger",
    "prob_naive_prior",
    "prob_logistic_regression",
    "prob_random_forest",
)


def top20_hit_rate(
    y_true: Any,
    y_score: Any,
    frac: float = 0.20,
) -> float:
    """Recall of positives inside the top ``ceil(frac * n)`` scores.

    Ties are broken by the smaller row index. This is the Phase A top-20% hit
    rate (share of charger cells captured), not precision inside that slice.
    """
    y = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    if y.shape[0] == 0 or y.shape[0] != scores.shape[0]:
        raise ValueError("y_true and y_score must be non-empty and the same length")
    n = int(y.shape[0])
    k = max(1, int(np.ceil(frac * n)))
    order = sorted(range(n), key=lambda i: (-float(scores[i]), i))
    hits = sum(int(y[i]) for i in order[:k])
    total_pos = int(np.sum(y))
    if total_pos == 0:
        return 0.0
    return float(hits) / float(total_pos)


def assert_leak_free_features(columns: list[str], manifest: dict[str, Any]) -> None:
    """Require ``columns`` to be exactly ``manifest['ml_features']`` and leak-free."""
    expected = [str(col) for col in manifest["ml_features"]]
    chosen = [str(col) for col in columns]
    if chosen != expected:
        raise ValueError(
            "feature_columns must equal feature_manifest.json ml_features exactly "
            f"(expected {expected}, got {chosen})"
        )
    excluded = {str(col) for col in manifest["excluded_target_and_audit"]}
    prefixes = tuple(str(prefix) for prefix in manifest["analysis_prefix_excluded"])
    for col in chosen:
        if col in excluded or col in _BANNED_EXACT or col.startswith(prefixes):
            raise ValueError(f"forbidden model input: {col}")
        if _EV_LEAK_RE.search(col):
            raise ValueError(f"forbidden model input: {col}")


def _reject_banned_feature_names(columns: list[str]) -> None:
    for col in columns:
        if col in _BANNED_EXACT or _EV_LEAK_RE.search(col):
            raise ValueError(f"forbidden model input: {col}")
        if col.startswith(("acc10_", "acc15_", "lisa_")):
            raise ValueError(f"forbidden model input: {col}")


def load_baseline_inputs(
    cells_path: Path,
    folds_path: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Load H3 r9 cells, check the frozen fold file, and return manifest features."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    features = [str(col) for col in manifest["ml_features"]]
    assert_leak_free_features(features, manifest)

    cells = gpd.read_file(cells_path)
    folds = pd.read_csv(folds_path)
    if "cell_id" not in cells.columns or "cell_id" not in folds.columns:
        raise ValueError("cell_id is required in cells and spatial_folds.csv")
    cells = cells.copy()
    cells["cell_id"] = cells["cell_id"].astype(str)
    folds = folds.copy()
    folds["cell_id"] = folds["cell_id"].astype(str)
    if cells["cell_id"].duplicated().any():
        raise ValueError("duplicate cell_id in cells")
    if folds["cell_id"].duplicated().any():
        raise ValueError("duplicate cell_id in spatial_folds.csv")
    if set(cells["cell_id"]) != set(folds["cell_id"]):
        raise ValueError("cell_id mismatch between cells and spatial_folds.csv")

    bad_res = [cid for cid in cells["cell_id"] if h3.get_resolution(cid) != 9]
    if bad_res:
        raise ValueError(f"expected H3 resolution 9 cells, found {bad_res[:3]}")

    fold_map = dict(zip(folds["cell_id"], folds["spatial_block"].astype(int), strict=True))
    mapped = cells["cell_id"].map(fold_map).to_numpy(dtype=int)
    observed = cells["spatial_block"].to_numpy(dtype=int)
    if not np.array_equal(mapped, observed):
        raise ValueError(
            "spatial_block mismatch between cells_base_res9.gpkg and spatial_folds.csv"
        )

    missing = [col for col in features if col not in cells.columns]
    if missing:
        raise ValueError(f"cells missing ml_features: {missing}")
    values = cells.loc[:, features].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("ML features contain NaN or Inf")
    return cells, features, manifest


def _round8(value: float) -> float:
    return float(np.round(float(value), 8))


def _score_block(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        roc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))
        brier = float(brier_score_loss(y_true, y_score))
    return {
        "roc_auc": _round8(roc),
        "pr_auc": _round8(pr_auc),
        "brier": _round8(brier),
        "top20_hit_rate": _round8(top20_hit_rate(y_true, y_score)),
    }


def _positive_proba(pipe: Pipeline, x: np.ndarray) -> np.ndarray:
    proba = pipe.predict_proba(x)
    classes = list(pipe.named_steps["clf"].classes_)
    if 1 not in classes:
        raise ValueError("classifier did not learn the positive class")
    return np.asarray(proba[:, classes.index(1)], dtype=float)


def _fit_dummy(x: np.ndarray, y: np.ndarray, seed: int) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", DummyClassifier(strategy="prior", random_state=seed)),
        ]
    )
    pipe.fit(x, y)
    return pipe


def _fit_logistic(x: np.ndarray, y: np.ndarray, blocks: np.ndarray, seed: int) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=4000,
                    random_state=seed,
                ),
            ),
        ]
    )
    search = GridSearchCV(
        estimator=pipe,
        param_grid={"clf__C": list(LOGISTIC_C_GRID)},
        cv=PredefinedSplit(np.asarray(blocks, dtype=int)),
        scoring="roc_auc",
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(x, y)
    best = search.best_estimator_
    if not isinstance(best, Pipeline):
        raise TypeError("logistic search did not return a Pipeline")
    return best


def _fit_forest(x: np.ndarray, y: np.ndarray, blocks: np.ndarray, seed: int) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=RF_N_ESTIMATORS,
                    max_features="sqrt",
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )
    search = GridSearchCV(
        estimator=pipe,
        param_grid={
            "clf__max_depth": list(RF_MAX_DEPTH_GRID),
            "clf__min_samples_leaf": list(RF_MIN_SAMPLES_LEAF_GRID),
        },
        cv=PredefinedSplit(np.asarray(blocks, dtype=int)),
        scoring="roc_auc",
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(x, y)
    best = search.best_estimator_
    if not isinstance(best, Pipeline):
        raise TypeError("random forest search did not return a Pipeline")
    return best


@dataclass
class BaselineReport:
    """Out-of-fold probabilities and the pipelines that produced them."""

    feature_columns: list[str]
    seed: int
    cell_id: list[str]
    y: np.ndarray
    spatial_block: np.ndarray
    oof_proba: dict[str, np.ndarray]
    fitted_pipelines: dict[str, dict[int, Pipeline]]

    @property
    def n_cells(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_pos(self) -> int:
        return int(np.sum(self.y))

    @property
    def prevalence(self) -> float:
        return float(self.n_pos) / float(self.n_cells)

    def published_oof(self) -> dict[str, np.ndarray]:
        """Probabilities rounded to 8 decimals, which is what the CSV stores."""
        return {
            name: np.round(np.asarray(self.oof_proba[name], dtype=float), 8) for name in MODEL_NAMES
        }

    def to_json_dict(self) -> dict[str, Any]:
        published = self.published_oof()
        blocks = np.asarray(self.spatial_block, dtype=int)
        models: dict[str, Any] = {}
        for name in MODEL_NAMES:
            probs = published[name]
            fold_metrics: list[dict[str, float | int]] = []
            for fold in range(5):
                mask = blocks == fold
                scored = _score_block(self.y[mask], probs[mask])
                fold_metrics.append({"fold": fold, **scored})
            block: dict[str, Any] = {"fold_metrics": fold_metrics}
            for metric in METRIC_NAMES:
                vals = np.asarray([row[metric] for row in fold_metrics], dtype=float)
                block[f"mean_{metric}"] = _round8(float(vals.mean()))
                block[f"std_{metric}"] = _round8(float(vals.std(ddof=0)))
            oof_scores = _score_block(self.y, probs)
            for metric in METRIC_NAMES:
                block[f"oof_{metric}"] = oof_scores[metric]
            models[name] = block
        return {
            "seed": int(self.seed),
            "n_cells": self.n_cells,
            "n_pos": self.n_pos,
            "prevalence": self.prevalence,
            "feature_columns": list(self.feature_columns),
            "models": models,
        }


def run_spatial_block_baselines(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    seed: int = 42,
) -> BaselineReport:
    """Fit dummy, logistic, and forest pipelines under frozen spatial blocks.

    The design matrix is ``feature_columns`` only. Scaler statistics and the
    inner ``PredefinedSplit`` search see training blocks and never the outer
    test block.
    """
    columns = [str(col) for col in feature_columns]
    _reject_banned_feature_names(columns)
    required = [*columns, "cell_id", "has_ev_charger", "spatial_block"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    x = frame.loc[:, columns].to_numpy(dtype=float, copy=True)
    if not np.isfinite(x).all():
        raise ValueError("ML features contain NaN or Inf")
    y_raw = frame["has_ev_charger"].to_numpy()
    if not np.isin(y_raw, [0, 1]).all():
        raise ValueError("has_ev_charger must be binary {0, 1}")
    y = y_raw.astype(int)
    blocks = frame["spatial_block"].to_numpy(dtype=int)
    if set(np.unique(blocks).tolist()) != {0, 1, 2, 3, 4}:
        raise ValueError("spatial_block must be the frozen blocks 0..4")
    for fold in range(5):
        y_fold = y[blocks == fold]
        if y_fold.size == 0 or int(y_fold.min()) == int(y_fold.max()):
            raise ValueError(f"spatial block {fold} must contain both classes")

    oof = {name: np.zeros(y.shape[0], dtype=float) for name in MODEL_NAMES}
    fitted: dict[str, dict[int, Pipeline]] = {name: {} for name in MODEL_NAMES}
    for fold in range(5):
        test = blocks == fold
        train = ~test
        x_train, y_train, blocks_train = x[train], y[train], blocks[train]
        x_test = x[test]
        models = {
            "naive_prior": _fit_dummy(x_train, y_train, seed),
            "logistic_regression": _fit_logistic(x_train, y_train, blocks_train, seed),
            "random_forest": _fit_forest(x_train, y_train, blocks_train, seed),
        }
        for name, pipe in models.items():
            fitted[name][fold] = pipe
            oof[name][test] = _positive_proba(pipe, x_test)

    return BaselineReport(
        feature_columns=columns,
        seed=int(seed),
        cell_id=frame["cell_id"].astype(str).tolist(),
        y=y,
        spatial_block=blocks,
        oof_proba=oof,
        fitted_pipelines=fitted,
    )


def write_baseline_artifacts(
    report: BaselineReport,
    metrics_path: Path,
    tables_dir: Path,
) -> dict[str, Path]:
    """Write the M3 JSON and the two CSV tables named in the M3 contract."""
    metrics_path = Path(metrics_path)
    tables_dir = Path(tables_dir)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    payload = report.to_json_dict()
    metrics_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    cv_rows: list[dict[str, Any]] = []
    for name in MODEL_NAMES:
        block = payload["models"][name]
        row: dict[str, Any] = {"model": name}
        for metric in METRIC_NAMES:
            row[f"mean_{metric}"] = block[f"mean_{metric}"]
            row[f"std_{metric}"] = block[f"std_{metric}"]
        for metric in METRIC_NAMES:
            row[f"oof_{metric}"] = block[f"oof_{metric}"]
        cv_rows.append(row)
    cv_path = tables_dir / "ml_cv_results.csv"
    pd.DataFrame(cv_rows, columns=list(CV_COLUMNS)).to_csv(cv_path, index=False)

    published = report.published_oof()
    oof_frame = pd.DataFrame(
        {
            "cell_id": report.cell_id,
            "spatial_block": np.asarray(report.spatial_block, dtype=int),
            "has_ev_charger": np.asarray(report.y, dtype=int),
            "prob_naive_prior": published["naive_prior"],
            "prob_logistic_regression": published["logistic_regression"],
            "prob_random_forest": published["random_forest"],
        }
    )
    oof_path = tables_dir / "ml_oof_predictions.csv"
    oof_frame.loc[:, list(OOF_COLUMNS)].to_csv(oof_path, index=False)
    return {"metrics": metrics_path, "cv": cv_path, "oof": oof_path}
