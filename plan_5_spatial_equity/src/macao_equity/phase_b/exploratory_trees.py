"""Exploratory boosted trees. These models are outside the pre-specified plan.

The outer split is the frozen spatial blocks. The grid search sees only the
other blocks. Reported scores use the same three metrics as Phase A and Route A.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from macao_equity.ml.baselines import assert_leak_free_features
from macao_equity.phase_b.metrics import mean_std, score_binary

MODEL_NAMES: tuple[str, ...] = ("xgboost", "lightgbm", "hist_gradient_boosting")
N_ESTIMATORS = 200

XGB_GRID: dict[str, tuple[Any, ...]] = {
    "clf__max_depth": (3, 6),
    "clf__learning_rate": (0.05, 0.1),
    "clf__min_child_weight": (1, 5),
}
LGBM_GRID: dict[str, tuple[Any, ...]] = {
    "clf__num_leaves": (15, 31),
    "clf__learning_rate": (0.05, 0.1),
    "clf__min_child_samples": (10, 20),
}
HGB_GRID: dict[str, tuple[Any, ...]] = {
    "clf__max_depth": (3, None),
    "clf__learning_rate": (0.05, 0.1),
    "clf__min_samples_leaf": (8, 20),
}
GRIDS: dict[str, dict[str, tuple[Any, ...]]] = {
    "xgboost": XGB_GRID,
    "lightgbm": LGBM_GRID,
    "hist_gradient_boosting": HGB_GRID,
}


def _estimator(name: str, seed: int, n_estimators: int) -> Pipeline:
    if name == "xgboost":
        clf: Any = XGBClassifier(
            n_estimators=n_estimators,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
        )
    elif name == "lightgbm":
        clf = LGBMClassifier(
            n_estimators=n_estimators,
            objective="binary",
            random_state=seed,
            n_jobs=1,
            verbose=-1,
            force_col_wise=True,
        )
    elif name == "hist_gradient_boosting":
        clf = HistGradientBoostingClassifier(
            max_iter=n_estimators,
            early_stopping=False,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown exploratory model: {name}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _fit_one(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    blocks_train: np.ndarray,
    seed: int,
    *,
    n_estimators: int,
    grids: dict[str, dict[str, tuple[Any, ...]]],
) -> tuple[Pipeline, dict[str, Any]]:
    """Grid search on training-block ids only. ``blocks_train`` must not contain the test id."""
    pipe = _estimator(name, seed, n_estimators)
    search = GridSearchCV(
        estimator=pipe,
        param_grid={key: list(values) for key, values in grids[name].items()},
        cv=PredefinedSplit(np.asarray(blocks_train, dtype=int)),
        scoring="roc_auc",
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    best = search.best_estimator_
    if not isinstance(best, Pipeline):
        raise TypeError(f"{name} search did not return a Pipeline")
    params = {}
    for key, value in search.best_params_.items():
        if isinstance(value, np.generic):
            value = value.item()
        params[str(key)] = value
    return best, params


def _positive_proba(pipe: Pipeline, x: np.ndarray) -> np.ndarray:
    proba = pipe.predict_proba(x)
    classes = list(pipe.named_steps["clf"].classes_)
    if 1 not in classes:
        raise ValueError("classifier did not learn the positive class")
    return np.asarray(proba[:, classes.index(1)], dtype=float)


def run_exploratory_trees(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    seed: int = 42,
    n_estimators: int = N_ESTIMATORS,
    grids: dict[str, dict[str, tuple[Any, ...]]] | None = None,
) -> dict[str, Any]:
    """Fit the three exploratory trees under the frozen five blocks."""
    columns = [str(col) for col in feature_columns]
    chosen_grids = grids if grids is not None else GRIDS
    x = frame.loc[:, columns].to_numpy(dtype=float, copy=True)
    if not np.isfinite(x).all():
        raise ValueError("ML features contain NaN or Inf")
    y = frame["has_ev_charger"].to_numpy()
    if not np.isin(y, [0, 1]).all():
        raise ValueError("has_ev_charger must be binary {0, 1}")
    y = y.astype(int)
    blocks = frame["spatial_block"].to_numpy(dtype=int)
    if set(np.unique(blocks).tolist()) != {0, 1, 2, 3, 4}:
        raise ValueError("spatial_block must be the frozen blocks 0..4")

    fold_rows: list[dict[str, Any]] = []
    for name in MODEL_NAMES:
        for fold in range(5):
            test = blocks == fold
            train = ~test
            if int(fold) in set(blocks[train].tolist()):
                raise RuntimeError("test block leaked into the training rows")
            pipe, params = _fit_one(
                name,
                x[train],
                y[train],
                blocks[train],
                seed,
                n_estimators=n_estimators,
                grids=chosen_grids,
            )
            scores = score_binary(y[test], _positive_proba(pipe, x[test]))
            fold_rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "label": "exploratory",
                    "best_params": params,
                    **scores,
                }
            )
    return {
        "label": "exploratory",
        "outside_prespecified_plan": True,
        "seed": int(seed),
        "n_estimators": int(n_estimators),
        "feature_columns": columns,
        "inner_tuning": "PredefinedSplit on the four training spatial blocks only",
        "inner_scoring": "roc_auc",
        "fold_rows": fold_rows,
    }


def _published_forest(phase_a_payload: dict[str, Any]) -> dict[str, float]:
    block = phase_a_payload["models"]["random_forest"]
    return {
        "mean_roc_auc": float(block["mean_roc_auc"]),
        "std_roc_auc": float(block["std_roc_auc"]),
        "mean_pr_auc": float(block["mean_pr_auc"]),
        "std_pr_auc": float(block["std_pr_auc"]),
        "mean_top20_hit_rate": float(block["mean_top20_hit_rate"]),
        "std_top20_hit_rate": float(block["std_top20_hit_rate"]),
    }


def summarize(
    report: dict[str, Any],
    phase_a_payload: dict[str, Any],
) -> dict[str, Any]:
    """Fold-mean and fold standard deviation, matching the Phase A summary."""
    rows = []
    forest = _published_forest(phase_a_payload)
    rows.append(
        {
            "model": "random_forest",
            "label": "published_phase_a",
            "n_seeds": 1,
            "std_definition": "ddof=0 across the 5 frozen folds; copied, not refit",
            **{key: forest[key] for key in forest},
        }
    )
    fold_frame = pd.DataFrame(report["fold_rows"])
    models: dict[str, Any] = {}
    for name in MODEL_NAMES:
        part = fold_frame[fold_frame["model"] == name]
        row: dict[str, Any] = {
            "model": name,
            "label": "exploratory",
            "n_seeds": 1,
            "std_definition": "ddof=0 across the 5 frozen folds; one seed",
        }
        for metric in ("roc_auc", "pr_auc", "top20_hit_rate"):
            mean, std = mean_std(part[metric].tolist(), ddof=0)
            row[f"mean_{metric}"] = mean
            row[f"std_{metric}"] = std
        models[name] = {
            **row,
            "fold_metrics": [
                {
                    "fold": int(item["fold"]),
                    "roc_auc": item["roc_auc"],
                    "pr_auc": item["pr_auc"],
                    "top20_hit_rate": item["top20_hit_rate"],
                    "best_params": item["best_params"],
                }
                for item in part.to_dict(orient="records")
            ],
        }
        rows.append(row)
    return {
        "label": "exploratory",
        "outside_prespecified_plan": True,
        "seed": report["seed"],
        "feature_columns": report["feature_columns"],
        "inner_tuning": report["inner_tuning"],
        "published_random_forest": forest,
        "models": models,
        "rows": rows,
    }


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Exploratory tree comparison",
        "",
        "These three models are outside the pre-specified plan.",
        "The random-forest row is copied from Phase A and was not refit.",
        "Tuning used only the four training spatial blocks.",
        "Metrics are ROC-AUC, PR-AUC, and the top-20% hit rate.",
        "± is the standard deviation across the five folds for seed 42.",
        "",
        "| 模型 | 标签 | ROC-AUC | PR-AUC | 前 20% 命中率 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {model} | {label} | {roc} | {pr} | {hit} |".format(
                model=row["model"],
                label=row["label"],
                roc=_fmt(row["mean_roc_auc"], row["std_roc_auc"]),
                pr=_fmt(row["mean_pr_auc"], row["std_pr_auc"]),
                hit=_fmt(row["mean_top20_hit_rate"], row["std_top20_hit_rate"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_exploratory_artifacts(
    summary: dict[str, Any],
    *,
    metrics_path: Path,
    markdown_path: Path,
) -> dict[str, Path]:
    metrics_path = Path(metrics_path)
    markdown_path = Path(markdown_path)
    for path in (metrics_path, markdown_path):
        if "colab_runs" in path.parts and "run1" in path.parts:
            raise RuntimeError(f"refusing to write archived run 1: {path}")
        if path.name.startswith("phase_b_route_b") or "run2" in path.name:
            raise RuntimeError(f"refusing to touch Route B run files: {path}")
        if path.name in {"phase_b_comparison.csv", "phase_b_comparison.md"}:
            raise RuntimeError(f"refusing to overwrite the comparison table: {path}")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(to_markdown(summary), encoding="utf-8")
    return {"metrics": metrics_path, "markdown": markdown_path}


def assert_exploratory_features(columns: list[str], manifest: dict[str, Any]) -> None:
    assert_leak_free_features(columns, manifest)
